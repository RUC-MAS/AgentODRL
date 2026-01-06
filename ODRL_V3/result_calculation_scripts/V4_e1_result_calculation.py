import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, TypedDict, Any
from collections import defaultdict

# --- Langchain & LangGraph 核心组件 ---
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# --- 导入您的真实模块 ---
from mongodb import MongoDBManager
from ODRL_Check import validate_odrl_against_shacl

# --- 用户需要配置的常量 ---

# MongoDB 配置
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "odrl3_final"
COLLECTION_NAME = "e1_41nano_2"

# OpenAI API Key (请确保环境变量 OPENAI_API_KEY 已设置)
# 或者取消下一行注释并提供文件路径
API_KEY_PATH = r"C:\Users\34085\Desktop\Agent\ALL_API_KEY1.txt"

# SHACL 文件路径 (真实路径)
SHACL_PATHS = {
    "set": r"ODRL_V3\data_preparation\shacl_for_validation\ODRL_Rule_Shapes.ttl",
    "offer": r"ODRL_V3\data_preparation\shacl_for_validation\ODRL_Offer_Shape.ttl",
    "agreement": r"ODRL_V3\data_preparation\shacl_for_validation\ODRL_Agreement_Shape.ttl",
}

# ODRL 类型对应的总约束数量
TOTAL_CONSTRAINTS_BY_TYPE = {
    "set": 29,
    "offer": 23,
    "agreement": 24,
}

# 结果输出文件路径
OUTPUT_RESULTS_PATH = Path(r"ODRL_V3\result\3e\e1\e1_41nano_V4_2.json")

# 定义需要评估的 ODRL 策略字段名 (与Mongo文档字段完全对应)
ODRL_STRATEGIES = {
    "ontology": 'initial_odrl',
    "vldb": 'final_odrl_branch_A_constraint',
    "vldb_semantic": 'enhanced_odrl_after_constraint',
    "semantic_syntactic": 'final_odrl_branch_B_validation'
}

# 新增：全局最大并发数量控制
# I/O密集型任务（如API调用）的并发限制
MAX_IO_CONCURRENCY = 50
io_semaphore = asyncio.Semaphore(MAX_IO_CONCURRENCY)

# CPU密集型任务（如SHACL验证）的并发限制
# 通常设置为机器的CPU核心数
MAX_CPU_CONCURRENCY = 20
cpu_semaphore = asyncio.Semaphore(MAX_CPU_CONCURRENCY)

async def return_zero() -> float:
    """一个简单的异步函数，用于在检测到无效ODRL时返回0.0分。"""
    return 0.0

# --- Pydantic 模型定义 ---
# 1. 为 "裁判1 (识别器)" 设计的新模型
class SemanticPointIdentification(BaseModel):
    """用于规范“裁判1”输出的数据结构。"""
    semantic_points: List[str] = Field(
        ...,
        description="从Policy文本中严格按照“语义点提取协议”提取出的、所有独立的原子语义点列表。"
    )

# --- 新增: "裁判2" 的精细化评估模型 ---
from enum import Enum  # <-- 新增此行

# --- 新增: "裁判2" 的精细化评估模型 ---
class EvaluationCategory(str, Enum):
    """定义语义评估的四个等级"""
    PERFECTLY_MATCHED = "PERFECTLY_MATCHED"
    PARTIALLY_MATCHED = "PARTIALLY_MATCHED"
    MISMATCHED = "MISMATCHED"
    MISSING = "MISSING"

class DetailedSemanticUnit(BaseModel):
    """对单个语义点的详细评估"""
    semantic_point_text: str = Field(..., description="从Policy文本中提取的原始语义点。")
    evaluation: EvaluationCategory = Field(..., description="该语义点在ODRL中的匹配程度评估。")
    justification: str = Field(..., description="对当前评估等级的简要解释。")

class FineGrainedSemanticEvaluation(BaseModel):
    """用于规范"裁判2"输出的精细化语义评估结果的数据结构"""
    evaluated_units: List[DetailedSemanticUnit] = Field(
        ...,
        description="对从裁判1提供的清单中的每一个语义点进行的详细评估列表。"
    )
    hallucinated_elements: List[str] = Field(
        default=[],
        description="在ODRL中存在，但无法在原始Policy文本中找到对应依据的元素描述列表（幻觉内容）。"
    )

# --- Prompt 定义 (复用自原代码) ---
IDENTIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
"""
您是一位精通ODRL和自然语言处理的专家。您的任务是精确评估给定的ODRL策略是否准确反映了原始Policy文本中的所有语义点。

**核心任务：**
1.  **识别语义点**: 严格遵循下述的 **“语义点提取协议”**，在脑中识别出`Policy 文本`中的所有原子语义单元。
2.  **验证映射**: 将识别出的每个语义点，逐一检查在`ODRL策略`中是否有对应且语义一致的表达。
3.  **量化结果**: 统计总语义点数量和在ODRL中被正确反映的数量，并列出未正确反映的语义点。

---
**语义点提取协议 (Semantic Unit Extraction Protocol)**

**1. 扫描锚点 (Scan Anchors):**
* **授权/接收实体**: 必须保留完整的实体名称 (例如, "the Urban Planning Dept")。
* **动作动词**: 必须包含所有修饰词和否定词 (例如, "securely process", "must not share")。
* **数据资产引用**: 必须包含相关的描述词 (例如, "Traffic Data")。
* **权限/禁止标记**: 例如, "grants", "prohibits"。
* **约束集群**: 必须将时间、目的、地点、数量等约束视为一个不可分割的整体 (例如, "until 2025-12-31", "except PII data")。
* **义务短语**: 例如, "must report", "is required to"。

**2. 提取规则 (Extraction Rules):**
* **✓ 忠实性 (Fidelity)**: 提取时必须保持原始文本的确切措辞、顺序和大小写。
* **✓ 完整性 (Completeness)**: 必须捕获完整的短语 (例如, 完整的日期 "2025-12-31")。
* **✓ 复合单元 (Compound Units)**: 必须保持逻辑单元的完整性 (例如, "must not process externally" 应被视为一个单元)。
* **✗ 排除项 (Exclusion)**: 必须忽略独立的、无实际语义的语法连接词 (如 that, which, a, the 等)。

**3. 验证示例 (Validation Examples):**
* **✓ 有效识别**
    * **文本**: "The Data Hub grants the Urban Planning Dept access to Traffic Data until 2025-12-31"
    * **应识别的语义点**: `["The Data Hub", "grants", "the Urban Planning Dept", "access", "Traffic Data", "until 2025-12-31"]`
* **✗ 无效识别**
    * **文本**: "Analytics teams may process sales records except PII data"
    * **错误识别**: `["Analytics", "may process", "sales records", "except", "PII data"]`
    * **正确识别**: `["Analytics teams", "may process", "sales records", "except PII data"]` (原因: "except PII data" 是一个不可分割的约束单元)。
---
**4. 输出要求:**
您的输出必须严格符合 `SemanticPointIdentification` 格式，只返回一个包含所有语义点字符串的 `semantic_points` 列表。不要添加任何额外的解释或评估。
"""),
    ("human",
    """
请根据“语义点提取协议”处理以下文本：

Policy 文本:
```{policy_text}```
""")
])

EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
"""
您是一位极其严苛和精准的ODRL分析专家，担任“裁判2”的角色。

---
**核心任务与评分协议 (Fine-Grained Scoring Protocol)**

**1. 评估范围:**
您的唯一任务是严格按照下方提供的`语义点清单`，逐项评估该清单中的每一个点在ODRL策略中的反映情况。`Policy 文本`仅供您在评估时理解上下文，但您评估的范围【必须】严格限定在给定的清单内。

**2. 逐点评估:**
对清单中的每一个语义点，您必须根据以下四个等级进行评估，并给出理由：

* **PERFECTLY_MATCHED (得分: 1.0):** ODRL中的表述完全、准确、无歧义地反映了该语义点的所有信息。
    * *示例*: 文本 "until 2025-12-31" 对应 ODRL `<odrl:constraint><odrl:rightOperand rdf:datatype="http://www.w3.org/2001/XMLSchema#date">2025-12-31</odrl:rightOperand>...</odrl:constraint>`。

* **PARTIALLY_MATCHED (得分: 0.5):** ODRL中反映了语义点的核心思想，但存在信息缺失或轻微不准确。
    * *示例*: 文本 "the Urban Planning Dept" 对应 ODRL `assignee: "Urban Dept"` (名称不完整)。文本 "securely process" 对应 ODRL `action: "process"` (缺少了“securely”这一修饰)。

* **MISMATCHED (得分: 0.1):** ODRL中有看似对应的元素，但其逻辑或含义与原文完全错误。
    * *示例*: 文本要求“允许(grants)”，ODRL中却使用了“禁止(prohibits)”。文本要求“目的为科研”，ODRL中却写成“目的为商业”。

* **MISSING (得分: 0.0):** ODRL中完全没有能对应上该语义点的任何信息。

**3. 幻觉内容检查:**
在完成清单评估后，检查ODRL中是否存在任何`Policy 文本`中未提及的限制、权限或实体（即“幻觉”内容）。

**4. 输出要求:**
您必须严格按照 `FineGrainedSemanticEvaluation` 格式输出结果。
- `evaluated_units` 列表的长度【必须】与输入的 `语义点清单` 的长度完全一致。
- `hallucinated_elements` 列表用于记录所有幻觉内容。
"""),
    ("human",
"""
请根据“精细化评分协议”，评估以下 ODRL 策略对“语义点清单”的反映情况。

Policy 文本 (供参考):
```{policy_text}```

语义点清单 (必须严格按照此清单逐一评估，不要增删):
```{semantic_points_list}```

ODRL 策略 (JSON-LD 格式):
```{odrl_policy_str}```
""")
])


# --- LangGraph 工作流状态定义 ---
class EvaluationState(TypedDict):
    """定义工作流中传递的状态"""
    mongo_uri: str
    db_name: str
    collection_name: str
    documents: List[Dict]
    # 修改: 'structured_llm' -> 'llm_clients'
    llm_clients: Dict[str, Any] # 将持有 'holistic' 和 'fine_grained' 两个客户端
    processed_results: List[Dict]
    final_aggregated_results: Dict



# --- LangGraph 节点函数 ---

def initialize_llm_clients():
    """初始化LLM和结构化输出链"""
    try:
        with open(API_KEY_PATH, "r", encoding='utf-8') as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        raise ValueError(f"错误：环境变量 'OPENAI_API_KEY' 未设置，也未在 {API_KEY_PATH} 找到文件。")
    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY 环境变量或在代码中提供 API_KEY_PATH")

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0, api_key=api_key, base_url="https://zzzzapi.com/v1")
    # 返回一个字典，包含两种新角色的客户端
    return {
        "identifier": llm.with_structured_output(SemanticPointIdentification),
        "evaluator": llm.with_structured_output(FineGrainedSemanticEvaluation)
    }

async def initialize_and_fetch_data(state: EvaluationState) -> EvaluationState:
    """节点1: 初始化LLM客户端，并从数据库获取数据"""
    print("--- 节点: 初始化 & 获取数据 ---")
    
    # 核心修正: 确保调用新的初始化函数，并将结果存入 'llm_clients' 键
    state['llm_clients'] = initialize_llm_clients()

    db_manager = MongoDBManager(mongo_uri=state['mongo_uri'], mongo_db_name=state['db_name'])

    # 定义投影以仅获取所需字段，提高效率
    projection_fields = {
        "usecase_key": 1,
        "usecase_text": 1,          # <-- 新增
        # "rewritten_usecase": 1,     # <-- 新增
        "policies.type": 1,
        "policies.text": 1,
        "policies.reflection_attempts_validation": 1,
    }
    for odrl_field in ODRL_STRATEGIES.values():
        projection_fields[f"policies.{odrl_field}"] = 1

    state['documents'] = await db_manager.fetch_all_rules(
        collection_name=state['collection_name'],
        projection=projection_fields
    )

    if not state['documents']:
        raise ValueError(f"从集合 '{state['collection_name']}' 未找到任何文档。请检查数据库和集合名称。")

    print(f"成功获取 {len(state['documents'])} 份 usecase 文档。")
    return state

async def run_syntax_validation_async(odrl_content: Dict, shacl_path: str, total_constraints: int) -> float:
    """
    在独立的线程中异步执行CPU密集的SHACL验证，并应用并发控制。
    """
    async with cpu_semaphore:
        try:
            # asyncio.to_thread 将同步函数放到一个单独的线程中运行，防止阻塞
            # validate_odrl_against_shacl 是一个同步函数
            _, _, num_violations, _ = await asyncio.to_thread(
                validate_odrl_against_shacl, odrl_content, shacl_path
            )
            score = (total_constraints - num_violations) / total_constraints
            return max(0, score)
        except Exception as e:
            print(f"语法验证失败，将返回 0.0 分。错误: {e}")
            return 0.0

async def run_semantic_identification_async(identifier_llm, policy_text: str) -> List[str]:
    """
    使用 "裁判1 (识别器)" 提取语义点列表。
    """
    async with io_semaphore:
        chain = IDENTIFICATION_PROMPT | identifier_llm
        try:
            result: SemanticPointIdentification = await chain.ainvoke({"policy_text": policy_text})
            return result.semantic_points
        except Exception as e:
            print(f"LLM 裁判1 (Identifier) 提取语义点失败: {e}")
            return [] # 返回空列表表示失败

async def run_fine_grained_evaluation_async(evaluator_llm, policy_text: str, semantic_points: List[str], odrl_policy_str: str) -> float:
    """
    使用 "裁判2 (评估器)" 对给定的语义点列表进行打分。
    """
    if not semantic_points: # 如果没有识别出语义点，则无法评估
        return 0.0

    async with io_semaphore:
        chain = EVALUATION_PROMPT | evaluator_llm
        try:
            result: FineGrainedSemanticEvaluation = await chain.ainvoke({
                "policy_text": policy_text,
                "semantic_points_list": "\n".join(f"- {p}" for p in semantic_points),
                "odrl_policy_str": odrl_policy_str
            })
            
            score_map = {
                EvaluationCategory.PERFECTLY_MATCHED: 1.0,
                EvaluationCategory.PARTIALLY_MATCHED: 0.5,
                EvaluationCategory.MISMATCHED: 0.1,
                EvaluationCategory.MISSING: 0.0,
            }
            
            # --- MODIFICATION START ---
            # 即使LLM返回的评估单元数不匹配，我们依然以输入的语义点总数为准来计算总分，这更公平。
            achieved_score = sum(score_map[unit.evaluation] for unit in result.evaluated_units)
            total_possible_score = len(semantic_points) # <-- 核心修正：分母永远是裁判1给出的数量
            
            if len(result.evaluated_units) != len(semantic_points):
                print(f"警告: 评估器返回的单元数({len(result.evaluated_units)})与裁判1识别的语义点数({len(semantic_points)})不匹配。将以识别点总数为准计算分数。")
            # --- MODIFICATION END ---
            
            base_score = achieved_score / total_possible_score if total_possible_score > 0 else 0.0
            
            # 应用幻觉惩罚
            hallucination_penalty = len(result.hallucinated_elements) * 0.2
            final_score = base_score - hallucination_penalty
            
            return max(0.0, final_score)
        except Exception as e:
            print(f"LLM 裁判2 (Evaluator) 评估失败: {e}")
            return 0.0

async def get_semantic_score(policy_text: str, odrl_content: Dict, llm_clients: Dict[str, Any]) -> float:
    """
    一个完整的语义评估流程：先识别，后评估。
    """
    # 步骤 1: 裁判1进行语义点识别
    semantic_points = await run_semantic_identification_async(llm_clients['identifier'], policy_text)
    
    # 步骤 2: 裁判2基于识别结果进行评估
    odrl_str = json.dumps(odrl_content)
    score = await run_fine_grained_evaluation_async(llm_clients['evaluator'], policy_text, semantic_points, odrl_str)
    
    return score

# ... (保留 run_syntax_validation_async, run_semantic_identification_async, 等辅助函数) ...

# --- NEW FUNCTION: Replaces evaluate_single_policy ---
async def evaluate_single_usecase(doc: Dict, llm_clients: Dict[str, Any]) -> Dict:
    """
    对单个Usecase的所有策略进行评估。
    - 语义评估：采用整体评估法，将所有ODRL视为一个整体。
    - 语法评估：对每个ODRL单独评分，然后取平均值。
    """
    usecase_key = doc['usecase_key']
    # 优先使用 "rewritten_usecase"，否则回退到 "usecase_text"
    usecase_text = doc.get("rewritten_usecase") or doc.get("usecase_text")
    
    if not usecase_text:
        print(f"警告: Usecase '{usecase_key}' 缺少 usecase_text 和 rewritten_usecase，无法进行语义评估。")
        # 返回一个包含usecase信息的空分数结构
        return {
            "usecase_key": usecase_key,
            "category": "unknown",
            "reflection_attempts": 0,
            "scores": {"syntactic": {s: 0.0 for s in ODRL_STRATEGIES}, "semantic": {s: 0.0 for s in ODRL_STRATEGIES}}
        }

    policies = doc.get("policies", [])
    usecase_scores = {"syntactic": {}, "semantic": {}}
    
    # --- 1. 整体语义评估 (Holistic Semantic Evaluation) ---
    # 首先，一次性从Usecase总文本中提取所有语义点
    semantic_points = await run_semantic_identification_async(llm_clients['identifier'], usecase_text)
    
    semantic_tasks = {}
    for strategy_name, odrl_field in ODRL_STRATEGIES.items():
        # 收集该策略下所有有效的 ODRL policies
        odrl_collection = [
            p.get(odrl_field) for p in policies 
            if p.get(odrl_field) and not isinstance(p.get(odrl_field), dict) or "error" not in p.get(odrl_field)
        ]
        
        if not odrl_collection:
            # 如果该策略没有生成任何有效ODRL，语义分为0
            semantic_tasks[strategy_name] = return_zero()
        else:
            # 将所有ODRL合并成一个JSON字符串（列表形式）
            odrl_str = json.dumps(odrl_collection)
            # 创建评估任务
            semantic_tasks[strategy_name] = run_fine_grained_evaluation_async(
                llm_clients['evaluator'], usecase_text, semantic_points, odrl_str
            )
            
    # 并发执行所有语义评估任务
    semantic_results = await asyncio.gather(*semantic_tasks.values())
    usecase_scores["semantic"] = dict(zip(semantic_tasks.keys(), semantic_results))

    # --- 2. 平均语法评估 (Averaged Syntactic Evaluation) ---
    # 语法评估仍需逐个policy进行，然后取平均
    syn_scores_by_strategy = defaultdict(list)
    syntax_tasks = []
    task_info = [] # 用于映射结果

    for policy in policies:
        odrl_type = policy.get("type")
        if odrl_type in TOTAL_CONSTRAINTS_BY_TYPE:
            total_constraints = TOTAL_CONSTRAINTS_BY_TYPE[odrl_type]
            shacl_path = SHACL_PATHS[odrl_type]
            for strategy_name, odrl_field in ODRL_STRATEGIES.items():
                odrl_content = policy.get(odrl_field)
                is_invalid = not odrl_content or (isinstance(odrl_content, dict) and "error" in odrl_content)
                
                task_info.append(strategy_name)
                if is_invalid:
                    syntax_tasks.append(return_zero())
                else:
                    syntax_tasks.append(run_syntax_validation_async(odrl_content, shacl_path, total_constraints))

    if syntax_tasks:
        syntax_results = await asyncio.gather(*syntax_tasks)
        for strategy, score in zip(task_info, syntax_results):
            syn_scores_by_strategy[strategy].append(score)

    for strategy_name in ODRL_STRATEGIES.keys():
        scores = syn_scores_by_strategy.get(strategy_name, [])
        usecase_scores["syntactic"][strategy_name] = sum(scores) / len(scores) if scores else 0.0

    # --- 3. 组装最终结果 ---
    category = "unknown"
    if usecase_key.startswith("su_"): category = "simple"
    elif usecase_key.startswith("cu_j_"): category = "concurrent"
    elif usecase_key.startswith("cu_p_"): category = "progressive"
    
    # 计算平均反思次数
    reflection_attempts = [p.get("reflection_attempts_validation", 0) for p in policies]
    avg_reflection = sum(reflection_attempts) / len(reflection_attempts) if reflection_attempts else 0

    return {
        "usecase_key": usecase_key,
        "category": category,
        "reflection_attempts": avg_reflection,
        "scores": usecase_scores
    }

# --- MODIFIED FUNCTION: Replaces the old evaluate_policies ---
async def evaluate_policies(state: EvaluationState) -> EvaluationState:
    """节点2: 并行评估所有 Usecases 的语法和语义得分"""
    print(f"\n--- 节点: 评估 Usecases (共 {len(state['documents'])} 个) ---")
    
    # 新逻辑: 对每个文档（Usecase）创建一个评估任务
    tasks = [evaluate_single_usecase(doc, state['llm_clients']) for doc in state['documents']]
    
    # 并发执行所有Usecase的评估
    processed_results = await asyncio.gather(*tasks)

    state['processed_results'] = processed_results
    print(f"评估完成，共处理 {len(processed_results)} 个 Usecases。")
    return state

# --- MODIFIED FUNCTION: Simplified aggregation logic ---
def aggregate_and_save_results(state: EvaluationState) -> EvaluationState:
    """节点3: 聚合所有得分并保存到文件"""
    print("\n--- 节点: 聚合结果 & 保存文件 ---")
    
    # 新的数据结构已经是Usecase级别的了，可以直接按类别聚合
    category_level_data = defaultdict(lambda: {"reflection_attempts": [], "syntactic": defaultdict(list), "semantic": defaultdict(list)})
    
    # 直接遍历处理好的Usecase结果
    for usecase_result in state['processed_results']:
        # 同时填充特定分类和'all'分类
        for category in [usecase_result['category'], 'all']: 
            category_level_data[category]['reflection_attempts'].append(usecase_result['reflection_attempts'])
            for s_name, score in usecase_result['scores']['syntactic'].items():
                category_level_data[category]['syntactic'][s_name].append(score)
            for s_name, score in usecase_result['scores']['semantic'].items():
                category_level_data[category]['semantic'][s_name].append(score)

    final_results = {}
    for category, data in category_level_data.items():
        final_results[category] = {"average_reflection_attempts": sum(data['reflection_attempts']) / len(data['reflection_attempts']) if data['reflection_attempts'] else 0, "performance_by_strategy": {}}
        for s_name in ODRL_STRATEGIES.keys():
            syn_scores, sem_scores = data['syntactic'][s_name], data['semantic'][s_name]
            final_results[category]["performance_by_strategy"][s_name] = {
                "syntactic_score": f"{sum(syn_scores) / len(syn_scores):.2%}" if syn_scores else "N/A",
                "semantic_score": f"{sum(sem_scores) / len(sem_scores):.2%}" if sem_scores else "N/A",
            }

    output_data = {"metadata": {"source_database": state['db_name'], "source_collection": state['collection_name']}, "results": final_results}

    # 确保输出目录存在
    OUTPUT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"聚合完成，结果已保存至: {OUTPUT_RESULTS_PATH}")
    state['final_aggregated_results'] = output_data
    return state

# --- 构建 LangGraph 工作流 ---
workflow = StateGraph(EvaluationState)
workflow.add_node("initialize_and_fetch", initialize_and_fetch_data)
workflow.add_node("evaluate_policies", evaluate_policies)
workflow.add_node("aggregate_and_save", aggregate_and_save_results)
workflow.set_entry_point("initialize_and_fetch")
workflow.add_edge("initialize_and_fetch", "evaluate_policies")
workflow.add_edge("evaluate_policies", "aggregate_and_save")
workflow.add_edge("aggregate_and_save", END)
app = workflow.compile()

# --- 主执行函数 ---
async def main():
    initial_state = {"mongo_uri": MONGO_URI, "db_name": MONGO_DB_NAME, "collection_name": COLLECTION_NAME}
    print("🚀 开始执行 LangGraph 工作流...")
    final_state = await app.ainvoke(initial_state)
    print("\n✅ 工作流执行完毕。")
    print("\n--- 最终评估结果 ---")
    print(json.dumps(final_state.get('final_aggregated_results', {}), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())