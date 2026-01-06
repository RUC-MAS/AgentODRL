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
COLLECTION_NAME = "e1_41"

# OpenAI API Key (请确保环境变量 OPENAI_API_KEY 已设置)
# 或者取消下一行注释并提供文件路径
API_KEY_PATH = r"C:\Users\34085\Desktop\Agent\ALL_API_KEY.txt"

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
OUTPUT_RESULTS_PATH = Path(r"ODRL_V3\result\3e\e1_41_r3.json")

# 定义需要评估的 ODRL 策略字段名 (与Mongo文档字段完全对应)
ODRL_STRATEGIES = {
    "ontology": 'initial_odrl',
    "vldb": 'final_odrl_branch_A_constraint',
    "vldb_semantic": 'enhanced_odrl_after_constraint',
    "semantic_syntactic": 'final_odrl_branch_B_validation'
}

# LLM 裁判团配置
LLM_JUDGE_COUNT = 2

# 新增：全局最大并发数量控制
# I/O密集型任务（如API调用）的并发限制
MAX_IO_CONCURRENCY = 50
io_semaphore = asyncio.Semaphore(MAX_IO_CONCURRENCY)

# CPU密集型任务（如SHACL验证）的并发限制
# 通常设置为机器的CPU核心数
MAX_CPU_CONCURRENCY = 8
cpu_semaphore = asyncio.Semaphore(MAX_CPU_CONCURRENCY)

async def return_zero() -> float:
    """一个简单的异步函数，用于在检测到无效ODRL时返回0.0分。"""
    return 0.0

# --- Pydantic 模型定义 (复用自原代码) ---
class SemanticEvaluation(BaseModel):
    """用于规范语义评估结果的数据结构，符合语义一致性得分计算逻辑"""
    total_semantic_units: int = Field(
        ...,
        description="从原始policy文本中分析出的所有独立语义点的总数量。每个语义点代表policy中的一个关键信息元素（实体、规则、条件、属性等），且不可再分"
    )
    correctly_reflected_units: int = Field(
        ...,
        description="在ODRL策略中被准确、完整表达出来的语义点的数量。准确表达指ODRL中有对应元素且语义一致"
    )
    missing_or_incorrect_units: List[str] = Field(
        ...,
        description="未被正确反映的语义点描述列表，说明哪些语义点在ODRL中缺失或表达错误"
    )

# --- Prompt 定义 (复用自原代码) ---
SEMANTIC_PROMPT = ChatPromptTemplate.from_messages([
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

**评估标准：**
* **正确反映**: ODRL中有直接对应元素且语义完全匹配。
* **未正确反映**: ODRL中缺失对应元素、元素语义不匹配或不完整。

**输出要求：**
请严格按照`SemanticEvaluation`格式返回结果，包含：
1.  `total_semantic_units`: 根据上述协议识别出的语义点总数。
2.  `correctly_reflected_units`: 在ODRL中被正确反映的语义点数量。
3.  `missing_or_incorrect_units`: 未被正确反映的语义点描述列表。
**注意：`total_semantic_units` 必须等于 `correctly_reflected_units` 与 `missing_or_incorrect_units` 列表长度之和。**
"""),
    ("human",
    """
请根据以下信息进行语义评估：

Policy 文本:
```{policy_text}```

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
    structured_llm: Any
    processed_results: List[Dict]
    final_aggregated_results: Dict


# --- LangGraph 节点函数 ---

def initialize_llm_client():
    """初始化LLM和结构化输出链"""
    try:
        with open(API_KEY_PATH, "r", encoding='utf-8') as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        raise ValueError(f"错误：环境变量 'OPENAI_API_KEY' 未设置，也未在 {API_KEY_PATH} 找到文件。")
    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY 环境变量或在代码中提供 API_KEY_PATH")

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1, api_key=api_key, base_url="https://zzzzapi.com/v1")
    return llm.with_structured_output(SemanticEvaluation)

async def initialize_and_fetch_data(state: EvaluationState) -> EvaluationState:
    """节点1: 初始化LLM客户端，并从数据库获取数据"""
    print("--- 节点: 初始化 & 获取数据 ---")
    state['structured_llm'] = initialize_llm_client()

    db_manager = MongoDBManager(mongo_uri=state['mongo_uri'], mongo_db_name=state['db_name'])

    # 定义投影以仅获取所需字段，提高效率
    projection_fields = {
        "usecase_key": 1,
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
    # 对于脚本类应用，通常不需要显式关闭连接
    # await db_manager.close_connection()
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

async def run_semantic_evaluation_async(structured_llm, policy_text: str, odrl_policy_str: str) -> float:
    """
    使用LLM计算单次语义得分（已添加并发控制）。
    """
    async with io_semaphore:
        chain = SEMANTIC_PROMPT | structured_llm
        try:
            result: SemanticEvaluation = await chain.ainvoke({
                "policy_text": policy_text,
                "odrl_policy_str": odrl_policy_str
            })
            return result.correctly_reflected_units / result.total_semantic_units if result.total_semantic_units > 0 else 0.0
        except Exception as e:
            print(f"LLM 语义评估失败，将返回 0.0 分。错误: {e}")
            return 0.0

async def evaluate_single_policy(policy: Dict, structured_llm: Any) -> Dict:
    """
    对单个policy的所有策略进行并行的语法和语义评估。
    新增：检查无效的ODRL（如包含错误信息），并直接将得分记为0。
    """
    policy_scores = {"syntactic": {}, "semantic": {}}
    syntax_tasks = {}
    semantic_tasks = {}

    # 1. 为所有需要计算的任务创建协程
    odrl_type = policy.get("type")
    if odrl_type in TOTAL_CONSTRAINTS_BY_TYPE:
        total_constraints = TOTAL_CONSTRAINTS_BY_TYPE[odrl_type]
        shacl_path = SHACL_PATHS[odrl_type]

        for strategy_name, odrl_field in ODRL_STRATEGIES.items():
            odrl_content = policy.get(odrl_field)

            # --- 新增的核心检查逻辑 ---
            # 如果 ODRL 内容为空，或是一个错误字典，则标记为无效
            is_invalid = not odrl_content or (isinstance(odrl_content, dict) and "error" in odrl_content)

            if is_invalid:
                # 当 ODRL 无效时，为其创建返回 0 的 "伪任务"
                # 这样可以保持后续 gather 和结果解析逻辑的统一性
                if odrl_content:  # 仅在有错误内容时打印日志，对于完全为空的不打印
                    print(f"检测到无效 ODRL [Policy Text: '{policy.get('text', '')[:30]}...', Strategy: {strategy_name}]. 得分记为 0。")
                
                syntax_tasks[strategy_name] = return_zero()
                semantic_tasks[strategy_name] = [return_zero() for _ in range(LLM_JUDGE_COUNT)]
            
            else:
                # ODRL 内容有效，按原逻辑创建评估任务
                # 创建语法任务
                syntax_tasks[strategy_name] = run_syntax_validation_async(
                    odrl_content, shacl_path, total_constraints
                )

                # 创建语义任务（每个裁判一个）
                odrl_str = json.dumps(odrl_content)
                semantic_tasks[strategy_name] = [
                    run_semantic_evaluation_async(structured_llm, policy["text"], odrl_str)
                    for _ in range(LLM_JUDGE_COUNT)
                ]

    # --- 后续逻辑保持不变 ---

    # 2. 并发执行所有已创建的任务
    all_tasks_to_run = list(syntax_tasks.values())
    for judge_tasks in semantic_tasks.values():
        all_tasks_to_run.extend(judge_tasks)

    # 如果没有任何任务，直接返回
    if not all_tasks_to_run:
        return policy_scores

    # 执行所有任务
    all_results = await asyncio.gather(*all_tasks_to_run)

    # 3. 解析结果
    result_index = 0
    # 解析语法得分
    for strategy_name in syntax_tasks.keys():
        policy_scores["syntactic"][strategy_name] = all_results[result_index]
        result_index += 1

    # 解析语义得分
    for strategy_name, judge_tasks in semantic_tasks.items():
        if not judge_tasks:
            continue

        # 提取属于当前策略的语义得分结果
        judge_scores = all_results[result_index : result_index + len(judge_tasks)]
        result_index += len(judge_tasks)

        # 计算平均分
        if judge_scores:
            policy_scores["semantic"][strategy_name] = sum(judge_scores) / len(judge_scores)

    return policy_scores

async def evaluate_policies(state: EvaluationState) -> EvaluationState:
    """节点2: 并行评估所有 policies 的语法和语义得分"""
    print(f"\n--- 节点: 评估 Policies (共 {len(state['documents'])} 个 Usecases) ---")
    tasks = [evaluate_single_policy(policy, state['structured_llm']) for doc in state['documents'] for policy in doc.get("policies", [])]
    all_policy_scores = await asyncio.gather(*tasks)

    processed_results = []
    policy_counter = 0
    for doc in state['documents']:
        usecase_key = doc['usecase_key']
        category = "unknown"
        if usecase_key.startswith("su_"): category = "simple"
        elif usecase_key.startswith("cu_j_"): category = "concurrent"
        elif usecase_key.startswith("cu_p_"): category = "progressive"

        for policy in doc.get("policies", []):
            processed_results.append({
                "usecase_key": usecase_key, "category": category,
                "reflection_attempts": policy.get("reflection_attempts_validation", 0),
                "scores": all_policy_scores[policy_counter]
            })
            policy_counter += 1

    state['processed_results'] = processed_results
    print(f"评估完成，共处理 {len(processed_results)} 个 policies。")
    return state

def aggregate_and_save_results(state: EvaluationState) -> EvaluationState:
    """节点3: 聚合所有得分并保存到文件"""
    print("\n--- 节点: 聚合结果 & 保存文件 ---")
    # 按 usecase_key -> category 聚合
    usecase_level_data = defaultdict(lambda: {"category": "", "reflection_attempts": [], "syntactic": defaultdict(list), "semantic": defaultdict(list)})
    for res in state['processed_results']:
        key = res['usecase_key']
        usecase_level_data[key]['category'] = res['category']
        usecase_level_data[key]['reflection_attempts'].append(res['reflection_attempts'])
        for s_name, score in res['scores']['syntactic'].items(): usecase_level_data[key]['syntactic'][s_name].append(score)
        for s_name, score in res['scores']['semantic'].items(): usecase_level_data[key]['semantic'][s_name].append(score)

    usecase_averages = {key: {"category": data['category'],
        "avg_reflection": sum(data['reflection_attempts']) / len(data['reflection_attempts']) if data['reflection_attempts'] else 0,
        "avg_syntactic": {s_name: sum(scores) / len(scores) for s_name, scores in data['syntactic'].items() if scores},
        "avg_semantic": {s_name: sum(scores) / len(scores) for s_name, scores in data['semantic'].items() if scores},
    } for key, data in usecase_level_data.items()}

    category_level_data = defaultdict(lambda: {"reflection_attempts": [], "syntactic": defaultdict(list), "semantic": defaultdict(list)})
    for key, avg_data in usecase_averages.items():
        for category in [avg_data['category'], 'all']: # 同时填充特定分类和'all'分类
            category_level_data[category]['reflection_attempts'].append(avg_data['avg_reflection'])
            for s_name, score in avg_data['avg_syntactic'].items(): category_level_data[category]['syntactic'][s_name].append(score)
            for s_name, score in avg_data['avg_semantic'].items(): category_level_data[category]['semantic'][s_name].append(score)

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