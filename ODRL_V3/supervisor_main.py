import asyncio
import json
import re
from typing import List, TypedDict, Dict, Any, Literal, Optional

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
# from langgraph.types import Send   # send+子图疑似异步表现接近同步，fw

from mongodb import MongoDBManager
from ODRL_Check import validate_odrl_against_shacl
import pypdf

import time
from datetime import datetime
from zoneinfo import ZoneInfo

from ODRL_V3.prompt.rewriter_prompt_V2 import rewriter_prompt
from ODRL_V3.prompt.splitter_prompt import splitter_prompt
from ODRL_V3.prompt.supervisor_prompt_V1 import supervisor_prompt

from log import TokenUsageCallbackHandler

# --- 1. LLM 初始化 (未改变) ---
# [新增] 实例化我们自定义的回调处理器
# [修改] 仅实例化回调处理器，但不在LLM初始化时附加
token_tracker = TokenUsageCallbackHandler()

API_KEY_PATH = r"C:\Users\34085\Desktop\Agent\ALL_API_KEY.txt"
with open(API_KEY_PATH, "r", encoding='utf-8') as f:
    api_key = f.read().strip()
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0, api_key=api_key, base_url="https://4zapi.com/v1/")
llm_for_utils = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1, api_key=api_key, base_url="https://4zapi.com/v1/")

# --- 用于语义点识别的 LoRA 模型 ---
# 重要：请根据实际情况修改这里的 URL 和模型路径
API_BASE_URL = "http://localhost:8000/v1" # 指向 vLLM 服务地址
MODEL_PATH = "/root/autodl-tmp/sft-qwen3-merged/" # vLLM 加载的模型路径

# 初始化 LangChain 的 ChatOpenAI 以连接到 LoRA 模型
# llm_lora_semantic = ChatOpenAI(
#     openai_api_base=API_BASE_URL,
#     model=MODEL_PATH,
#     openai_api_key="not-needed", # vLLM 不需要 key
#     temperature=0.2,
#     extra_body={"chat_template_kwargs": {"enable_thinking": False}}
# )

llm_lora_semantic = llm

# --- 2. 配置 (未改变) ---
LOG_FILE_PATH = r"ODRL_V3\result\log\test.json"

USECASE_JSON_PATH = r"ODRL_V3\data_preparation\simple_usecases copy 2.json"
SET_ONTOLOGY_PATH = r"ODRL_V3\data_preparation\ontology_for_prompt\ODRL_Rule_Generator_template.pdf"
OFFER_ONTOLOGY_PATH = r"ODRL_V3\data_preparation\ontology_for_prompt\ODRL-Offer_Generator_Template.pdf"
AGREEMENT_ONTOLOGY_PATH = r"ODRL_V3\data_preparation\ontology_for_prompt\ODRL-Agreement_Generator_template.pdf"
SET_SHACL_PATH = r"ODRL_V3\data_preparation\shacl_for_validation\ODRL_Rule_Shapes.ttl"
OFFER_SHACL_PATH = r"ODRL_V3\data_preparation\shacl_for_validation\ODRL_Offer_Shape.ttl"
AGREEMENT_SHACL_PATH = r"ODRL_V3\data_preparation\shacl_for_validation\ODRL_Agreement_Shape.ttl"
SET_CONSTRAINT_PATH = r"ODRL_V3\data_preparation\constraint_for_prompt\set_constraints.txt"
OFFER_CONSTRAINT_PATH = r"ODRL_V3\data_preparation\constraint_for_prompt\offer_constraints.txt"
AGREEMENT_CONSTRAINT_PATH = r"ODRL_V3\data_preparation\constraint_for_prompt\agreement_constraints.txt"
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "test"
MONGO_COLLECTION_NAME = "e3_41nano_split_6" # 使用新的集合名称以作区分
MAX_REFLECTIONS = 3

# [核心修改] 移除了 ENABLE_REFERENCE_RESOLUTION 和 ENABLE_POLICY_SPLITTING 全局开关
# 它们的执行现在由 Supervisor 动态决定

# [新增] 全局开关，用于控制是否执行基于约束的反射分支 (分支A)
ENABLE_CONSTRAINT_REFLECTION_BRANCH = False

MAX_CONCURRENCY = 60 # 控制并发任务数量

SUPERVISOR_RECURSION_LIMIT = 40

# --- 3. Pydantic 输出模板 (大部分未改变) ---
# [新增]：链式引用解析结果模型
class RewriterAnalysisResult(BaseModel):
    is_chained: bool = Field(description="是否存在条款间的链式引用关系。")
    rewritten_text: str = Field(description="如果存在引用，这是将引用内容替换展开后的新文本；否则与原始文本相同。")
    reference_graph: Optional[Dict[str, List[str]]] = Field(default_factory=dict, description="一个表示引用关系的图，键为引用条款，值为被引用的条款列表。")
class PolicyClassification(BaseModel):
    policy_type: Literal["set", "offer", "agreement"] = Field(description="The type of the policy, must be 'set', 'offer', or 'agreement'.")
    policy_text: str = Field(description="The exact segment of text from the usecase corresponding to this policy.")

class Policies(BaseModel):
    policies: List[PolicyClassification] = Field(description="A list of all policies identified in the usecase text.")

# [核心修改] 新增 Supervisor 的决策模型
class SupervisorDecision(BaseModel):
    """Supervisor 的决策模型，定义了下一步要执行的动作。"""
    next_action: Literal["call_rewriter", "call_splitter", "call_generator"] = Field(
        description=(
            "基于当前用例状态和历史记录的决策。"
            " 'call_rewriter': 当文本包含引用或递进关系时调用。"
            " 'call_splitter': 当文本复杂，需要拆分为多个策略时调用。"
            " 'call_generator': 当文本已准备好，可以进行最终 ODRL 生成时调用。这是最终步骤。"
        )
    )

# --- 4. 辅助函数 (未改变) ---
def clean_json_string(s: str) -> str:
    match = re.search(r"```(json)?\s*([\s\S]*?)\s*```", s)
    return match.group(2).strip() if match else s

async def load_content(path: str) -> str:
    if path.endswith(".pdf"):
        with open(path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            return "\n".join(page.extract_text() for page in reader.pages)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

# --- 5. LangGraph 工作流状态与节点定义 ---

# 状态定义: 为子图和主图分别定义状态
class PolicyInfo(TypedDict):
    # 此字典将携带单个策略从开始到结束的所有产出
    type: Literal["set", "offer", "agreement"]
    text: str
    reflection_attempts: int
    # 各阶段产物
    initial_odrl: Optional[Dict[str, Any]]
    semantic_points_from_lora: Optional[str] # 新增: 存储LoRA模型提取的语义点
    enhanced_odrl: Optional[Dict[str, Any]]
    # 分支A产物
    final_odrl_branch_A_constraint: Optional[Dict[str, Any]]
    # 分支B产物
    enhanced_odrl_after_constraint: Optional[Dict[str, Any]]
    final_odrl_branch_B_validation: Optional[Dict[str, Any]]

class SubgraphState(TypedDict):
    # 子图的状态，处理单个 usecase
    usecase_key: str
    original_usecase_text: str # [新增] 保存原始文本
    policies: List[PolicyInfo]

    # [新增] 引用解析相关字段
    is_chained: bool
    rewritten_usecase: Optional[str]
    reference_graph: Optional[Dict]

    # [核心修改] 新增字段，用于接收 supervisor 的决策历史
    history: List[str]
    
    # 用于在分支合并时传递数据
    policies_branch_A_done: Optional[List[PolicyInfo]]
    policies_branch_B_done: Optional[List[PolicyInfo]]
    # 子图的最终产物
    final_document_for_usecase: Optional[Dict[str, Any]]

# [核心修改] 新增 Supervisor 工作流的状态
class SupervisorState(TypedDict):
    """管理单个 use case 在 supervisor 决策循环中的状态。"""
    usecase_key: str
    original_usecase_text: str
    current_usecase_text: str # 被 rewriter 或 splitter 修改的文本
    
    # 记录 supervisor 的决策历史
    history: List[str]
    
    # Rewriter 的产物
    is_chained: bool
    reference_graph: Optional[Dict]
    
    # Splitter 的产物，准备传递给 generator 子图
    policies_for_generator: Optional[List[PolicyInfo]]
    
    # Supervisor 的最新决策
    supervisor_decision: Optional[SupervisorDecision]


# --- [核心修改] Supervisor 和重构后的 Agent 节点 ---

# [新增] Supervisor 节点
async def supervisor_node(state: SupervisorState) -> Dict[str, Any]:
    """
    Supervisor Agent 的核心节点。
    分析当前 use case 的状态，并决定下一个动作。
    """
    print(f"\n--- SUPERVISOR evaluating Usecase: '{state['usecase_key']}' ---")
    print(f"  History: {state['history']}")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", supervisor_prompt),
        ("user", "Usecase Original Text: {usecase_text}\n\nDicision Action History：{history}\n\nCurrent Usecase Text:\n```\n{current_text}\n```\n\n请做出你的下一步决策。")
    ])
    
    chain = prompt | llm.with_structured_output(SupervisorDecision)
    
    decision = await chain.ainvoke({
        "usecase_text": state['original_usecase_text'],
        "history": state['history'],
        "current_text": state['current_usecase_text']
    }, config={"callbacks": [token_tracker]})
    
    print(f"  Supervisor Decision: '{decision.next_action}'")
    
    return {"supervisor_decision": decision}

# [新增] Rewriter 节点 (从原 chain_reference_resolver_node 改造)
async def rewriter_node(state: SupervisorState) -> Dict[str, Any]:
    """
    执行引用关系解析和重写。
    """
    print(f"--- AGENT: Rewriter working on '{state['usecase_key']}' ---")
    
    state['current_usecase_text'] = state['original_usecase_text'] # 将reset状态的逻辑分配到这

    rewriter_prompt = ChatPromptTemplate.from_messages([
        ("system", rewriter_prompt),
        ("user", "Usecase Text:\n\n{usecase}")
    ])
    rewriter_chain = rewriter_prompt | llm_for_utils.with_structured_output(RewriterAnalysisResult)
    
    result = await rewriter_chain.ainvoke({"usecase": state['current_usecase_text']}, config={"callbacks": [token_tracker]})
    
    # 更新状态
    history = state['history'] + ["call_rewriter"]
    return {
        "current_usecase_text": result.rewritten_text,
        "is_chained": result.is_chained,
        "reference_graph": result.reference_graph,
        "history": history,
        "policies_for_generator": [PolicyInfo(type="set", text=result.rewritten_text, reflection_attempts=0)]
    }

# Splitter 节点 
async def splitter_node(state: SupervisorState) -> Dict[str, Any]:
    """
    将 use case 文本拆分为多个策略。
    """
    print(f"--- AGENT: Splitter working on '{state['usecase_key']}' ---")
    
    if state['history'] and state['history'][-1] == "call_splitter":
        state['current_usecase_text'] = state['original_usecase_text']
 
    splitter_prompt = ChatPromptTemplate.from_messages([
        ("system", splitter_prompt),
        ("user", "Analyze the following usecase text:\n\n{usecase}")
    ])
    splitter_chain = splitter_prompt | llm_for_utils.with_structured_output(Policies)
    
    result = await splitter_chain.ainvoke({"usecase": state['current_usecase_text']}, config={"callbacks": [token_tracker]})
    
    policies_for_generator = [
        PolicyInfo(type=p.policy_type, text=p.policy_text, reflection_attempts=0)
        for p in result.policies
    ]

    new_current_text = "\n".join([f"  - Splitted_Policy {i+1} ({p['type']}): '{p['text']}'" for i, p in enumerate(policies_for_generator)])
    
    history = state['history'] + ["call_splitter"]
    return {
        "policies_for_generator": policies_for_generator,
        "history": history,
        "current_usecase_text": new_current_text # 使用纯净的、组合后的文本更新状态
    }

    # # [新增] 创建一个代表拆分后状态的新文本
    # if len(policies_for_generator) > 1:
    #     summary_texts = [f"  - Policy {i+1} ({p['type']}): '{p['text'][:75]}...'" for i, p in enumerate(policies_for_generator)]
    #     new_current_text = (
    #         "The use case has been successfully split into the following policies:\n"
    #         + "\n".join(summary_texts)
    #         + "\n\nThe content is now ready for generation."
    #     )
    # elif len(policies_for_generator) == 1:
    #     new_current_text = f"The use case has been processed and confirmed as a single policy of type '{policies_for_generator[0]['type']}'. It is ready for generation."
    # else:
    #     new_current_text = "Splitter did not identify any policies. This may be an error."
    
    # history = state['history'] + ["call_splitter"]

# # [新增] Reset 节点
# async def reset_state_node(state: SupervisorState) -> Dict[str, Any]:
#     """
#     将 use case 状态重置为初始状态。
#     """
#     print(f"--- ACTION: Resetting state for '{state['usecase_key']}' ---")
#     return {
#         "current_usecase_text": state['original_usecase_text'],
#         "history": state['history'] + ["reset_state"],
#         "is_chained": False,
#         "reference_graph": None,
#         "policies_for_generator": None
#     }

# [新增 & 核心修改] Generator 调用节点
async def run_generator_subgraph_node(state: SupervisorState) -> None:
    """
    准备数据并调用 ODRL 生成子图 (generator)。
    """
    print(f"--- SUPERVISOR: Handing off '{state['usecase_key']}' to Generator Subgraph ---")
    
    # [修改] 简化逻辑：我们现在假定 supervisor 在调用 generator 之前，
    # 必须已经调用过 splitter 来获取策略列表（即使列表长度为1）。
    # 因此，不再需要处理 policies_for_generator 为空的情况。
    if not state.get('policies_for_generator'):
        # 这是一个健壮性检查，理论上不应该发生
        print(f"ERROR: 'call_generator' was triggered for '{state['usecase_key']}' but splitter has not been run. Aborting generator step.")
        policies = [PolicyInfo(type="set", text=state['current_usecase_text'], reflection_attempts=0)]

    else:
        policies = state['policies_for_generator']

    # 准备子图的初始状态
    subgraph_input_state = SubgraphState(
        usecase_key=state['usecase_key'],
        original_usecase_text=state['original_usecase_text'],
        policies=policies,
        is_chained=state.get('is_chained', False),
        rewritten_usecase=state['current_usecase_text'] if state.get('is_chained') else None,
        reference_graph=state.get('reference_graph'),
        # [修改] 将 supervisor 的历史记录传递给子图
        history=state['history']
    )
    
    # 异步调用子图
    await subgraph.ainvoke(subgraph_input_state)
    print(f"--- GENERATOR SUBGRAPH for '{state['usecase_key']}' finished. ---")
    return

# [新增] Supervisor 的条件路由函数
def supervisor_router(state: SupervisorState) -> str:
    """根据 supervisor 的决策，决定下一个节点的走向。"""
    decision = state.get("supervisor_decision")
    if not decision:
        return "supervisor" # 如果没有决策，回到 supervisor
    return decision.next_action


# --- 子图节点 (这部分代码几乎没有改动，因为 supervisor 不关心 generator 的内部实现) ---
# 节点的 prompt 和 chain 定义在函数外部，以便复用
_initial_odrl_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert in ODRL. Convert the natural language policy into ODRL JSON-LD.

**Crucial Formatting Rules:**
1.  Output ONLY the raw JSON-LD content. Do not use markdown fences or add explanations.
2.  When a value has a specific datatype (like a date or a typed string), you MUST use the JSON-LD expanded object format.
3.  **NEVER use the `^^` syntax.** This is invalid in JSON.

**Correct Example for typed values:**
{{
  "dct:issued": {{
    "@value": "2023-10-04T00:00:00Z",
    "@type": "xsd:dateTime"
  }},
  "dct:description": {{
    "@value": "A sample description.",
    "@type": "xsd:string"
  }}
}}

**Incorrect Example (DO NOT DO THIS):**
{{
  "dct:issued": "2023-10-04T00:00:00Z"^^xsd:dateTime
}}
"""),
    ("user", "The policy type is: {policy_type}\n\nPolicy Text: {policy_text}\n\nOntology Info:\n{ontology}\n\nPlease generate the ODRL JSON-LD.")
])


_initial_odrl_chain = _initial_odrl_prompt | llm | StrOutputParser()

async def generate_initial_odrl_node(state: SubgraphState) -> Dict[str, Any]:
    print(f"--- Subgraph for '{state['usecase_key']}': Generating Initial ODRL ---")
    ontology_map = {
        "set": await load_content(SET_ONTOLOGY_PATH),
        "offer": await load_content(OFFER_ONTOLOGY_PATH),
        "agreement": await load_content(AGREEMENT_ONTOLOGY_PATH)
    }
    for i, policy in enumerate(state['policies']):
        ontology_content = ontology_map[policy['type']]
        odrl_str = await _initial_odrl_chain.ainvoke({
            "policy_type": policy['type'], 
            "policy_text": policy['text'], 
            "ontology": ontology_content
        },config={"callbacks": [token_tracker]})
        try:
            state['policies'][i]['initial_odrl'] = json.loads(clean_json_string(odrl_str))
        except json.JSONDecodeError:
            state['policies'][i]['initial_odrl'] = {"error": "LLM output was not valid JSON", "raw_output": odrl_str}
    return {"policies": state['policies']}

# 语义增强节点逻辑 (已修改)
async def semantic_enhancement_node(state: SubgraphState) -> Dict[str, Any]:
    """此节点已被重构，以使用专门微调的LoRA模型进行语义点提取。"""
    print(f"--- Subgraph for '{state['usecase_key']}': Semantic Enhancement ---")
    
    # 定义调用LoRA模型的prompt和chain
    lora_prompt_template = [
        ("system", "从给定的规则文本中提取所有核心语义点。"),
        ("user", "{policy_text}"),
    ]
    lora_chain = (
        ChatPromptTemplate.from_messages(lora_prompt_template) 
        | llm_lora_semantic 
        | StrOutputParser()
    )

    # 定义用于ODRL增强的prompt和chain（已适配LoRA模型输出）
    enhancer_prompt_template = [
        ("system", "You are an ODRL semantic alignment expert. Your task is to revise an 'Initial ODRL' to ensure it fully incorporates all the 'Key Semantic Points' extracted from the original 'Rule Text (RS)'. Your Process: 1. Read the 'Rule Text (RS)'. 2. Read the 'Initial ODRL' to understand the current representation. 3. Read the 'Key Semantic Points'. 4. Modify the 'Initial ODRL' JSON to incorporate any missing information from the semantic points and correct any misrepresentations. 5. Ensure the revised ODRL is a complete and accurate representation of the 'Rule Text (RS)'. 6. The final output must be only the raw, revised, and valid JSON-LD content. Do not add explanations. Adhere to all JSON-LD formatting rules, especially for typed values."),
        ("user", "Rule Text (RS):\n{rs_text}\n\nKey Semantic Points (extracted by a specialized model):\n{semantic_points}\n\nInitial ODRL (to be fixed):\n{initial_odrl}\n\nPlease generate the enhanced ODRL JSON-LD that covers all key semantic points.")
    ]
    enhancer_chain = (
        ChatPromptTemplate.from_messages(enhancer_prompt_template) 
        | llm 
        | StrOutputParser()
    )

    for policy in state['policies']:
        # 初始化新字段
        policy['semantic_points_from_lora'] = ""

        if not policy.get('initial_odrl') or "error" in policy['initial_odrl']:
            policy['enhanced_odrl'] = policy.get('initial_odrl', {"error": "Skipped enhancement due to invalid initial ODRL"})
            continue
        
        # 步骤 1: 使用 LoRA 模型提取语义点
        print(f"INFO: Extracting semantic points for policy in '{state['usecase_key']}' with LoRA model...")
        semantic_points_str = await lora_chain.ainvoke({"policy_text": policy['text']}, config={"callbacks": [token_tracker]})
        policy['semantic_points_from_lora'] = semantic_points_str
        
        # 步骤 2: 基于提取出的语义点，请求GPT增强初始ODRL
        initial_odrl_str = json.dumps(policy['initial_odrl'], indent=2, ensure_ascii=False)
        print(f"INFO: Enhancing ODRL for policy in '{state['usecase_key']}' using extracted points...")
        
        enhanced_odrl_str = await enhancer_chain.ainvoke({
            "rs_text": policy['text'], 
            "semantic_points": semantic_points_str, 
            "initial_odrl": initial_odrl_str
        }, config={"callbacks": [token_tracker]})
        try:
            policy['enhanced_odrl'] = json.loads(clean_json_string(enhanced_odrl_str))
        except json.JSONDecodeError:
            policy['enhanced_odrl'] = {"error": f"Semantic enhancement failed: LLM output invalid JSON", "raw_output": enhanced_odrl_str}

    return {"policies": state['policies']}

# --- 反射节点 (可重用逻辑) ---
_constraint_reflection_prompt = prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a meticulous ODRL JSON editor. Your task is to revise an initial ODRL policy based on a set of constraints. You must operate by editing the initial JSON, not by creating new nested structures.

**Your process MUST follow these steps:**
1.  **Analyze the initial ODRL:** Use the provided `Initial ODRL` JSON as the starting point and base for your work.
2.  **Apply Constraints:** Carefully read the `Must-follow Constraints` and the original `Rule Text (RS)`. Modify, add, or remove attributes and objects *within the existing structure* of the `Initial ODRL` to ensure it complies with all constraints and accurately reflects the rule.
3.  **Maintain Structure:** The final output MUST be a single, coherent ODRL JSON object. **DO NOT** create new, arbitrary, nested top-level objects. Edit the policy in-place.
4.  **Enforce Syntax:** Ensure the final output is perfectly valid JSON-LD.
    
**Crucial Formatting Rules:**
1.  Output ONLY the raw JSON-LD content. Do not use markdown fences or add explanations.
2.  When a value has a specific datatype (like a date or a typed string), you MUST use the JSON-LD expanded object format.
3.  **NEVER use the `^^` syntax.** This is invalid in JSON.

**Correct Example for typed values:**
{{
  "dct:issued": {{
    "@value": "2023-10-04T00:00:00Z",
    "@type": "xsd:dateTime"
  }},
  "dct:description": {{
    "@value": "A sample description.",
    "@type": "xsd:string"
  }}
}}

**Incorrect Example (DO NOT DO THIS):**
{{
  "dct:issued": "2023-10-04T00:00:00Z"^^xsd:dateTime
}}
"""),
    ("user", "Rule Text (RS): {rs_text}\n\nInitial ODRL:\n{initial_odrl}\n\nMust-follow Constraints:\n{constraint_info}\n\nPlease generate the revised ODRL based on the constraints.")
])
_constraint_reflection_chain = _constraint_reflection_prompt | llm | StrOutputParser()

async def _run_constraint_reflection(policies: List[PolicyInfo], input_key: str, output_key: str) -> List[PolicyInfo]:
    constraint_map = {
        "set": await load_content(SET_CONSTRAINT_PATH),
        "offer": await load_content(OFFER_CONSTRAINT_PATH),
        "agreement": await load_content(AGREEMENT_CONSTRAINT_PATH)
    }

    # [修改] 我们需要判断当前是否在为分支B工作
    # 最简单的方法是检查输入键
    is_branch_b = (input_key == 'enhanced_odrl')
    for policy in policies:
        odrl_to_reflect = policy.get(input_key)
        if not odrl_to_reflect or "error" in odrl_to_reflect:
            policy[output_key] = odrl_to_reflect or {"error": f"Skipped constraint reflection due to no valid input from '{input_key}'"}
            continue
        
        invoke_config = {"callbacks": [token_tracker]} if is_branch_b else {}
        corrected_odrl_str = await _constraint_reflection_chain.ainvoke({
            "rs_text": policy['text'], 
            "initial_odrl": json.dumps(odrl_to_reflect, indent=2), 
            "constraint_info": constraint_map[policy['type']]
        },
        config=invoke_config # <--- [新增]
    )
        try:
            policy[output_key] = json.loads(clean_json_string(corrected_odrl_str))
        except json.JSONDecodeError:
            policy[output_key] = {"error": "LLM output was not valid JSON", "raw_output": corrected_odrl_str}
    return policies

# [新增] 用于子图的条件分支路由器
def branch_router(state: SubgraphState):
    """根据全局开关决定工作流的分支路径。"""
    if ENABLE_CONSTRAINT_REFLECTION_BRANCH:
        print("INFO (Subgraph Router): Branch A (Constraint Reflection) is ENABLED. Executing both branches.")
        # 如果开启，则同时执行两个分支
        return ["reflect_on_initial_with_constraints", "semantic_enhancement"]
    else:
        print("INFO (Subgraph Router): Branch A (Constraint Reflection) is DISABLED. Skipping to Branch B.")
        # 如果关闭，则仅执行分支B
        return ["semantic_enhancement"]

async def reflect_on_initial_with_constraints_node(state: SubgraphState) -> Dict[str, Any]:
    """分支A: 直接在初始ODRL上进行基于约束的反射"""
    print(f"--- Subgraph for '{state['usecase_key']}': Running Branch A (Constraint Reflection on Initial) ---")
    updated_policies = await _run_constraint_reflection(
        policies=state['policies'], 
        input_key='initial_odrl',
        output_key='final_odrl_branch_A_constraint'
    )
    return {"policies_branch_A_done": updated_policies}

async def reflect_on_enhanced_with_constraints_node(state: SubgraphState) -> Dict[str, Any]:
    """分支B Part 1: 在增强ODRL上进行基于约束的反射"""
    print(f"--- Subgraph for '{state['usecase_key']}': Running Branch B, Step 1 (Constraint Reflection on Enhanced) ---")
    updated_policies = await _run_constraint_reflection(
        policies=state['policies'],
        input_key='enhanced_odrl',
        output_key='enhanced_odrl_after_constraint'
    )
    return {"policies": updated_policies}


_validation_reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an ODRL and SHACL expert. Reflect on the validation errors and revise the ODRL policy to be correct and accurately represent the rule text (RS). Output only the final, revised JSON-LD content."),
        ("user", "RS: {rs_text}\nInitial ODRL: {initial_odrl}\nValidation Result: {validation_tool_output}\nPlease generate the revised ODRL in JSON-LD format.")
    ])
_validation_reflection_chain = _validation_reflection_prompt | llm | StrOutputParser()

async def reflect_on_enhanced_with_validation_node(state: SubgraphState) -> Dict[str, Any]:
    """分支B Part 2: 在(约束反射后的)增强ODRL上进行基于SHACL验证的反射"""
    print(f"--- Subgraph for '{state['usecase_key']}': Running Branch B, Step 2 (Validation Reflection) ---")
    shacl_path_map = {"set": SET_SHACL_PATH, "offer": OFFER_SHACL_PATH, "agreement": AGREEMENT_SHACL_PATH}
    
    for policy in state['policies']:
        # 输入来自B分支上一步的产物
        odrl_to_validate = policy.get('enhanced_odrl_after_constraint')
        if not odrl_to_validate or "error" in odrl_to_validate:
            policy['final_odrl_branch_B_validation'] = odrl_to_validate or {"error": "Skipped validation due to no valid input"}
            continue

        current_odrl_dict = odrl_to_validate
        policy['reflection_attempts'] = 0
        while policy['reflection_attempts'] < MAX_REFLECTIONS:
            current_odrl_str = json.dumps(current_odrl_dict, indent=2)
            _, __, num_violations, validation_report = validate_odrl_against_shacl(odrl_content_str=current_odrl_str, shacl_ttl_path=shacl_path_map[policy['type']])
            
            if num_violations == 0:
                print(f"INFO (Validation): Policy in '{state['usecase_key']}' passed validation after {policy['reflection_attempts']} attempts.")
                break
            
            policy['reflection_attempts'] += 1
            validation_output_for_llm = f"Violations: {num_violations}\nErrors:\n{json.dumps(validation_report, ensure_ascii=False, indent=2)}"
            
            corrected_odrl_str = await _validation_reflection_chain.ainvoke({"rs_text": policy['text'], "initial_odrl": current_odrl_str, "validation_tool_output": validation_output_for_llm},config={"callbacks": [token_tracker]})
            try:
                current_odrl_dict = json.loads(clean_json_string(corrected_odrl_str))
            except json.JSONDecodeError:
                print(f"ERROR (Validation): LLM output for policy in '{state['usecase_key']}' was not valid JSON. Breaking loop.")
                current_odrl_dict = {"error": "LLM output not valid JSON after reflection", "raw_output": corrected_odrl_str}
                break
        
        policy['final_odrl_branch_B_validation'] = current_odrl_dict
        
    return {"policies_branch_B_done": state['policies']}


# --- 合并与存储节点 ---
async def prepare_document_node(state: SubgraphState) -> Dict[str, Any]:
    """子图合并节点: 合并活动分支的结果，为当前usecase准备最终文档"""
   # 分支B是主线，我们总是等待它
    policies_from_B = state.get('policies_branch_B_done')
    if not policies_from_B:
         print(f"INFO for '{state['usecase_key']}': Waiting for Branch B to complete...")
         return {}

    # 如果分支A被启用，我们才等待它
    if ENABLE_CONSTRAINT_REFLECTION_BRANCH and not state.get('policies_branch_A_done'):
        print(f"INFO for '{state['usecase_key']}': Branch A is enabled. Waiting for it to complete...")
        return {}
        
    print(f"--- Subgraph for '{state['usecase_key']}': Merging Active Branches and Preparing Document ---")
    
    # 使用 .get() 安全地获取分支A的结果，因为它可能不存在
    policies_from_A = state.get('policies_branch_A_done')

    policy_results_for_db = []
    # 以分支B的结果为主体进行合并
    for i, p_b in enumerate(policies_from_B):
        def safe_json_dumps(data):
            return json.dumps(data, ensure_ascii=False, indent=2) if isinstance(data, dict) else "{}"

        policy_doc = {
            "type": p_b['type'],
            "text": p_b['text'],
            "semantic_points_from_lora": p_b.get('semantic_points_from_lora', ''),
            "reflection_attempts_validation": p_b.get('reflection_attempts', 0),
            "initial_odrl": safe_json_dumps(p_b.get('initial_odrl')),
            "enhanced_odrl": safe_json_dumps(p_b.get('enhanced_odrl')),
            "enhanced_odrl_after_constraint": safe_json_dumps(p_b.get('enhanced_odrl_after_constraint')),
            "final_odrl_branch_B_validation": safe_json_dumps(p_b.get('final_odrl_branch_B_validation')),
        }

        # 仅当分支A被执行时，才合并其结果
        if policies_from_A and len(policies_from_A) > i:
            p_a = policies_from_A[i]
            policy_doc["final_odrl_branch_A_constraint"] = safe_json_dumps(p_a.get('final_odrl_branch_A_constraint'))
        else:
            # 否则，明确记录该分支被跳过
            policy_doc["final_odrl_branch_A_constraint"] = json.dumps({"status": "skipped_by_config"})

        policy_results_for_db.append(policy_doc)
    
    # 构建要存储的最终文档
    document = {
        "usecase_key": state['usecase_key'],
        "usecase_text": state['original_usecase_text'], # 存储最原始的文本
        "supervisor_decision_history": state['history'], # [核心修改] 将决策历史添加到最终文档中
        "policies": policy_results_for_db # 这是您已有的合并后的策略列表
    }
    
    # 根据state中的信息，添加引用解析相关的字段
    if state.get("is_chained"):
        document["rewritten_usecase"] = state["rewritten_usecase"]
        document["reference_graph"] = state.get("reference_graph")
        print(f"INFO: Adding rewritten usecase to the database document for '{state['usecase_key']}'.")

    return {"final_document_for_usecase": document}

async def save_document_to_db_node(state: SubgraphState) -> None:
    """子图最终节点: 将单个处理完成的文档存入数据库"""
    print(f"--- Subgraph for '{state['usecase_key']}': Saving document to Database ---")
    document_to_save = state.get("final_document_for_usecase")
    
    if not document_to_save: # 等待prepare_document_node整理两个分支要存入的数据
        print(f"WARNING: No document to save for usecase '{state['usecase_key']}'.")
        return None

    db_manager = MongoDBManager(mongo_uri=MONGO_URI, mongo_db_name=MONGO_DB_NAME)
    try:
        inserted_id = await db_manager.insert_generated_odrl(
            collection_name=MONGO_COLLECTION_NAME,
            odrl_data=document_to_save
        )
        print(f"SUCCESS: Saved document for '{state['usecase_key']}' with id {inserted_id} to collection '{MONGO_COLLECTION_NAME}'.")
    finally:
        await db_manager.close_connection()
    return None


# --- 6. 构建与运行工作流 ---

# Step 1: 定义子图 (这部分保持不变, 它其实是 Generator Agent)
policy_processor_graph = StateGraph(SubgraphState)
# 添加子图节点
policy_processor_graph.add_node("generate_initial_odrl", generate_initial_odrl_node)
# 分支A
policy_processor_graph.add_node("reflect_on_initial_with_constraints", reflect_on_initial_with_constraints_node)
# 分支B
policy_processor_graph.add_node("semantic_enhancement", semantic_enhancement_node)
policy_processor_graph.add_node("reflect_on_enhanced_with_constraints", reflect_on_enhanced_with_constraints_node)
policy_processor_graph.add_node("reflect_on_enhanced_with_validation", reflect_on_enhanced_with_validation_node)
# 合并节点
policy_processor_graph.add_node("prepare_document", prepare_document_node)
# 存储节点
policy_processor_graph.add_node("save_document_to_db", save_document_to_db_node)


# 设置子图流程 (这部分保持不变)
policy_processor_graph.set_entry_point("generate_initial_odrl")

# [修改] 使用条件边替代原来的直接分叉
policy_processor_graph.add_conditional_edges(
    "generate_initial_odrl",
    branch_router,
    {
        "reflect_on_initial_with_constraints": "reflect_on_initial_with_constraints",
        "semantic_enhancement": "semantic_enhancement"
    }
)

policy_processor_graph.add_edge("semantic_enhancement", "reflect_on_enhanced_with_constraints")
policy_processor_graph.add_edge("reflect_on_enhanced_with_constraints", "reflect_on_enhanced_with_validation")
policy_processor_graph.add_edge("reflect_on_initial_with_constraints", "prepare_document")
policy_processor_graph.add_edge("reflect_on_enhanced_with_validation", "prepare_document")
policy_processor_graph.add_edge("prepare_document", "save_document_to_db")
policy_processor_graph.set_finish_point("save_document_to_db")

# 编译子图
subgraph = policy_processor_graph.compile()


# --- [核心修改] Step 2: 定义新的、以 Supervisor 为核心的主图 ---
supervisor_workflow = StateGraph(SupervisorState)

# 添加 Supervisor 和 Agent 节点
supervisor_workflow.add_node("supervisor", supervisor_node)
supervisor_workflow.add_node("call_rewriter", rewriter_node)
supervisor_workflow.add_node("call_splitter", splitter_node)
# supervisor_workflow.add_node("reset_state", reset_state_node)
supervisor_workflow.add_node("call_generator", run_generator_subgraph_node)

# 设置主图流程
supervisor_workflow.set_entry_point("supervisor")

# 添加从 Agent 返回到 Supervisor 的边
supervisor_workflow.add_edge("call_rewriter", "supervisor")
supervisor_workflow.add_edge("call_splitter", "supervisor")
# supervisor_workflow.add_edge("reset_state", "supervisor")

# 添加 Supervisor 的条件路由
supervisor_workflow.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "call_rewriter": "call_rewriter",
        "call_splitter": "call_splitter",
        # "reset_state": "reset_state",
        "call_generator": "call_generator"
    }
)

# Generator 是终点
supervisor_workflow.add_edge("call_generator", END)

# 编译主图
app = supervisor_workflow.compile()


# [新增] 用于保存运行结果和配置的函数
def save_run_log(log_path: str, execution_time: float, token_handler: TokenUsageCallbackHandler):
    """收集配置、时间和token消耗，并追加到JSON日志文件中。"""
    # 动态获取模型名称，以防未来更改
    models_used = {
        "main_llm": llm.model_name,
        "utils_llm": llm_for_utils.model_name,
    }
    # 检查lora模型是否存在
    if 'llm_lora_semantic' in globals():
        models_used["lora_llm"] = llm_lora_semantic.model_name
        
    run_config = {
        "models_used": models_used,
        "usecase_source": USECASE_JSON_PATH,
        "mongo_collection": MONGO_COLLECTION_NAME,
        "max_concurrency": MAX_CONCURRENCY,
        "max_reflections": MAX_REFLECTIONS,
        "switches": {
            # [核心修改] 更新日志记录的开关信息
            "SUPERVISOR_ENABLED": True,
            "ENABLE_CONSTRAINT_REFLECTION_BRANCH": ENABLE_CONSTRAINT_REFLECTION_BRANCH
        }
    }
    
    run_results = {
        "total_execution_time_seconds": round(execution_time, 4),
        "tracked_token_usage": {
            "prompt_tokens": token_handler.prompt_tokens,
            "completion_tokens": token_handler.completion_tokens,
            "total_tokens": token_handler.total_tokens
        }
    }

    log_entry = {
        "timestamp_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
        "configuration": run_config,
        "results": run_results
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []
    
    logs.append(log_entry)
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)
        
    print(f"\n--- Run log saved to {log_path} ---")

# --- [核心修改] 运行逻辑 ---
async def main():
    # 1. 加载所有 use cases
    with open(USECASE_JSON_PATH, 'r', encoding='utf-8') as f:
        usecases = json.load(f)

    # 2. 为每个 use case 创建初始状态
    initial_states = [
        SupervisorState(
            usecase_key=key,
            original_usecase_text=text,
            current_usecase_text=text,
            history=[],
            is_chained=False,
            reference_graph=None,
            policies_for_generator=None,
            supervisor_decision=None
        )
        for key, text in usecases.items()
    ]

    print(f"--- Starting Supervisor-led workflow for {len(initial_states)} usecases in parallel ---")

    # 3. 并行执行所有 use case 的 supervisor 工作流
    # [新增] 创建一个信号量来限制并发
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    
    async def run_one_workflow(state: SupervisorState):
        async with semaphore:
            # ainvoke 会异步地完整运行 supervisor 驱动的图
            return await app.ainvoke(state,config={"recursion_limit": SUPERVISOR_RECURSION_LIMIT})

    tasks = [run_one_workflow(state) for state in initial_states]
    await asyncio.gather(*tasks)

    print(f"\n--- Main Workflow Finished ---")

if __name__ == "__main__":
    start_time = time.perf_counter()
    
    # [重要] 确保回调处理器在运行前被重置
    # 这一步现在更加关键，因为它不再在每次调用时自动重置
    token_tracker.reset()

    asyncio.run(main())
    
    elapsed = time.perf_counter() - start_time
    print(f"\nTOTAL EXECUTION TIME: {elapsed:.4f} seconds")

    # [修改] 打印最终的token统计信息
    print(f"\n--- Token Usage Summary ---")
    print(f"Total Prompt Tokens: {token_tracker.prompt_tokens}")
    print(f"Total Completion Tokens: {token_tracker.completion_tokens}")
    print(f"Total Tokens: {token_tracker.total_tokens}")


    # [修改] 在所有任务结束后，调用保存函数
    save_run_log(LOG_FILE_PATH, elapsed, token_tracker)