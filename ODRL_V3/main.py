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

from prompt.rewriter_prompt_V2 import rewriter_prompt
from prompt.splitter_prompt import splitter_prompt

from log import TokenUsageCallbackHandler

# --- 1. LLM 初始化 ---
# 仅实例化回调处理器，但不在LLM初始化时附加
token_tracker = TokenUsageCallbackHandler()

API_KEY_PATH = r"C:\Users\34085\Desktop\Agent\ALL_API_KEY2.txt"
with open(API_KEY_PATH, "r", encoding='utf-8') as f:
    api_key = f.read().strip()
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0, api_key=api_key, base_url="https://4zapi.com/v1/" )
llm_for_utils = ChatOpenAI(model="gpt-4.1", temperature=0.1, api_key=api_key, base_url="https://4zapi.com/v1/")

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

llm_lora_semantic = ChatOpenAI(
    model="gpt-5-nano",
    openai_api_key=api_key,
    temperature=0.2,
    base_url="https://4zapi.com/v1/"
)

# --- 2. 配置 ---
LOG_FILE_PATH = r"ODRL_V3\result\log\temp.json"

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
MONGO_DB_NAME = "odrl202601_temp"
MONGO_COLLECTION_NAME = "for_demo"

MAX_REFLECTIONS = 4

# [新增] 全局开关，用于控制是否执行引用解析节点
ENABLE_REFERENCE_RESOLUTION = False 
# --- 新增: 是否启用 usecase 分割器的开关 ---
ENABLE_POLICY_SPLITTING = True # 设置为 False 可将每个usecase视为单一策略
# [新增] 全局开关，用于控制是否执行基于约束的反射分支 (分支A)
ENABLE_CONSTRAINT_REFLECTION_BRANCH = False

MAX_CONCURRENCY = 60 # 控制send数量

# --- 3. Pydantic 输出模板 ---
# 链式引用解析结果模型
class ReferenceAnalysisResult(BaseModel):
    is_chained: bool = Field(description="是否存在条款间的链式引用关系。")
    rewritten_text: str = Field(description="如果存在引用，这是将引用内容替换展开后的新文本；否则与原始文本相同。")
    reference_graph: Optional[Dict[str, List[str]]] = Field(default_factory=dict, description="一个表示引用关系的图，键为引用条款，值为被引用的条款列表。")
class PolicyClassification(BaseModel):
    policy_type: Literal["set", "offer", "agreement"] = Field(description="The type of the policy, must be 'set', 'offer', or 'agreement'.")
    policy_text: str = Field(description="The exact segment of text from the usecase corresponding to this policy.")

class Policies(BaseModel):
    policies: List[PolicyClassification] = Field(description="A list of all policies identified in the usecase text.")

# --- 仅用于在不分割策略时确定策略类型的模型 ---
class JustThePolicyType(BaseModel):
    policy_type: Literal["set", "offer", "agreement"] = Field(description="The type of the policy, must be 'set', 'offer', or 'agreement'.")

# --- 4. 辅助函数 ---
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
    semantic_points_from_lora: Optional[str] # 存储LoRA模型提取的语义点
    enhanced_odrl: Optional[Dict[str, Any]]
    # 分支A产物
    final_odrl_branch_A_constraint: Optional[Dict[str, Any]]
    # 分支B产物
    enhanced_odrl_after_constraint: Optional[Dict[str, Any]]
    final_odrl_branch_B_validation: Optional[Dict[str, Any]]

class SubgraphState(TypedDict):
    # 子图的状态，处理单个 usecase
    usecase_key: str
    original_usecase_text: str # 保存原始文本
    usecase_text_for_processing: str # 即usecase_text
    policies: List[PolicyInfo]

    # 引用解析相关字段
    is_chained: bool
    rewritten_usecase: Optional[str]
    reference_graph: Optional[Dict]
    
    # 用于在分支合并时传递数据
    policies_branch_A_done: Optional[List[PolicyInfo]]
    policies_branch_B_done: Optional[List[PolicyInfo]]
    # 子图的最终产物
    final_document_for_usecase: Optional[Dict[str, Any]]

class MainGraphState(TypedDict):
    # 主图的状态，处理所有 usecase
    usecase_inputs: List[Dict[str, str]]
    # 用于在主图节点间传递处理后的usecase数据
    resolved_texts: List[Dict[str, Any]] 
    split_usecases: List[SubgraphState] # 存储所有拆分好的usecase，用于分发

# --- 节点逻辑 ---
# 独立的、可开关的引用解析节点
async def chain_reference_resolver_node(state: MainGraphState) -> Dict[str, Any]:
    """
    工作流的入口节点。加载所有usecases，并根据全局开关，
    并行地进行链式引用解析。
    """
    if not ENABLE_REFERENCE_RESOLUTION:
        print("--- INFO: Chained Reference Resolution is DISABLED. Skipping this step. ---")
        # 如果禁用，则直接加载并以默认值传递
        usecase_path = state['usecase_inputs'][0]['path']
        with open(usecase_path, 'r', encoding='utf-8') as f:
            usecases = json.load(f)
        
        resolved_texts = [
            {
                "key": key,
                "original_text": text,
                "text_for_processing": text, # 未处理
                "is_chained": False,
                "rewritten_usecase": None,
                "reference_graph": {}
            }
            for key, text in usecases.items()
        ]
        return {"resolved_texts": resolved_texts}

    print("--- Starting Workflow: Chained Reference Resolution (in parallel) ---")
    usecase_path = state['usecase_inputs'][0]['path']
    with open(usecase_path, 'r', encoding='utf-8') as f:
        usecases = json.load(f)

    resolver_prompt = ChatPromptTemplate.from_messages([
        ("system", rewriter_prompt),
        ("user", "Usecase Text:\n\n{usecase}")
    ])
    resolver_chain = resolver_prompt | llm_for_utils.with_structured_output(ReferenceAnalysisResult)

    # 创建一个信号量来限制并发
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _resolve_one_usecase(key: str, text: str) -> Dict[str, Any]:
         # 在执行任务前获取信号量
        async with semaphore:
            print(f"INFO: Starting reference analysis for '{key}'...")
            result = await resolver_chain.ainvoke({"usecase": text},config={"callbacks": [token_tracker]})
            print(f"INFO: Reference analysis for '{key}' complete. Chained: {result.is_chained}")
            return {
                "key": key,
                "original_text": text,
                "text_for_processing": result.rewritten_text,
                "is_chained": result.is_chained,
                "rewritten_usecase": result.rewritten_text if result.is_chained else None,
                "reference_graph": result.reference_graph
            }

    tasks = [_resolve_one_usecase(key, text) for key, text in usecases.items()]
    resolved_texts = await asyncio.gather(*tasks)
    
    return {"resolved_texts": resolved_texts}
# 节点 1: 加载并拆分所有 Usecases (已修改为并行处理)
async def load_and_split_all_usecases_node(state: MainGraphState) -> Dict[str, Any]:
    print("--- Starting Workflow: Loading and Splitting All Usecases (in parallel) ---")
    usecases_to_process = state.get("resolved_texts", [])
    if not usecases_to_process:
        return {"split_usecases": []}
    
    tasks = []
    # 创建一个信号量来限制并发
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # --- 基于全局开关决定执行逻辑 ---
    if ENABLE_POLICY_SPLITTING:
        print("INFO: Policy splitting is ENABLED. Processing usecases in parallel.")
        prompt = ChatPromptTemplate.from_messages([
        ("system", splitter_prompt),
        ("user", "Analyze the following usecase text:\n\n{usecase}")
    ])
        chain = prompt | llm_for_utils.with_structured_output(Policies)

        # 定义一个内部异步函数来处理单个usecase的拆分
        async def _split_one_usecase(usecase_data: Dict[str, Any]) -> SubgraphState:
            # 在执行任务前获取信号量
            async with semaphore:
                text_to_split = usecase_data["text_for_processing"]
                result = await chain.ainvoke({"usecase": text_to_split}, config={"callbacks": [token_tracker]})
                policies_with_state = [
                    PolicyInfo(type=p.policy_type, text=p.policy_text, reflection_attempts=0)
                    for p in result.policies
                ]
                # 传递所有信息到子图状态
                return SubgraphState(
                    usecase_key=usecase_data["key"],
                    original_usecase_text=usecase_data["original_text"],
                    usecase_text_for_processing=text_to_split,
                    policies=policies_with_state,
                    is_chained=usecase_data["is_chained"],
                    rewritten_usecase=usecase_data["rewritten_usecase"],
                    reference_graph=usecase_data["reference_graph"]
                )
        
        # 为每个usecase创建一个异步任务
        tasks = [_split_one_usecase(data) for data in usecases_to_process]

    else:
        # 当不分割时，将每个usecase作为一个整体，仅判断其类型
        print("INFO: Policy splitting is DISABLED. Processing usecases in parallel.")
        type_only_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an ODRL policy analysis expert. Your only task is to analyze a natural language rule text and determine its ODRL policy type.
            Apply this logic sequentially:
            1.  Test for: Agreement - A binding, executed contract between a specific Assigner and a specific Assignee.
            2.  Test for: Offer - A proposal from a specific Assigner to a generic class of potential Assignees.
            3.  Default to: Set - The policy defines generic rules and does not meet the criteria for an Agreement or Offer.
            Output your answer in the requested JSON format.
            """),
            ("user", "Analyze the following usecase text and determine its policy type:\n\n{usecase}")
        ])
        type_chain = type_only_prompt | llm_for_utils.with_structured_output(JustThePolicyType)

        # 定义一个内部异步函数来处理单个usecase的类型判断
        async def _type_one_usecase(usecase_data: Dict[str, Any]) -> SubgraphState:
            # 在执行任务前获取信号量
            async with semaphore:
                text_to_type = usecase_data["text_for_processing"]
                type_result = await type_chain.ainvoke({"usecase": text_to_type}, config={"callbacks": [token_tracker]})
                # 将整个usecase文本视为一个策略
                policies_with_state = [
                    PolicyInfo(
                        type=type_result.policy_type,
                        text=text_to_type, 
                        reflection_attempts=0
                    )
                ]
                # 同样传递所有信息到子图状态
                return SubgraphState(
                    usecase_key=usecase_data["key"],
                    original_usecase_text=usecase_data["original_text"],
                    usecase_text_for_processing=text_to_type,
                    policies=policies_with_state,
                    is_chained=usecase_data["is_chained"],
                    rewritten_usecase=usecase_data["rewritten_usecase"],
                    reference_graph=usecase_data["reference_graph"]
                )
        tasks = [_type_one_usecase(data) for data in usecases_to_process]

    split_usecases = await asyncio.gather(*tasks)
    print(f"INFO: All {len(split_usecases)} usecases have been processed for policies.")

    return {"split_usecases": split_usecases}

# --- 子图节点 ---
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

# 语义增强节点逻辑
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
        
        # 步骤 2: 基于提取出的语义点，请求llm增强初始ODRL
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

# --- 反射节点 ---
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

    # 需要判断当前是否在为分支B工作, 最简单的方法是检查输入键
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
        config=invoke_config 
    )
        try:
            policy[output_key] = json.loads(clean_json_string(corrected_odrl_str))
        except json.JSONDecodeError:
            policy[output_key] = {"error": "LLM output was not valid JSON", "raw_output": corrected_odrl_str}
    return policies

# 用于子图的条件分支路由器
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
        "policies": policy_results_for_db # 这是已有的合并后的策略列表
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

# Step 1: 定义子图
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


# 设置子图流程
policy_processor_graph.set_entry_point("generate_initial_odrl")

# 使用条件边替代直接分叉
policy_processor_graph.add_conditional_edges(
    "generate_initial_odrl",
    branch_router,
    {
        "reflect_on_initial_with_constraints": "reflect_on_initial_with_constraints",
        "semantic_enhancement": "semantic_enhancement"
    }
)

# policy_processor_graph.add_edge("generate_initial_odrl", "reflect_on_initial_with_constraints") # -> 分支A
# policy_processor_graph.add_edge("generate_initial_odrl", "semantic_enhancement") # -> 分支B
policy_processor_graph.add_edge("semantic_enhancement", "reflect_on_enhanced_with_constraints")
policy_processor_graph.add_edge("reflect_on_enhanced_with_constraints", "reflect_on_enhanced_with_validation")
policy_processor_graph.add_edge("reflect_on_initial_with_constraints", "prepare_document")
policy_processor_graph.add_edge("reflect_on_enhanced_with_validation", "prepare_document")
policy_processor_graph.add_edge("prepare_document", "save_document_to_db")
policy_processor_graph.set_finish_point("save_document_to_db")

# 编译子图
subgraph = policy_processor_graph.compile()


# 并行执行节点，替代原有的 dispatcher_edge 和 Send 机制
async def run_subgraphs_in_parallel_node(state: MainGraphState):
    """
    使用 asyncio.gather 并发执行所有子图。
    """
    usecases_to_process = state.get("split_usecases", [])
    if not usecases_to_process:
        print("INFO: No usecases to process. Ending workflow.")
        return

    print(f"\nINFO: Concurrently executing {len(usecases_to_process)} subgraphs using asyncio.gather...")

    # 创建一个信号量来限制并发
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    async def run_one_subgraph(usecase_state: SubgraphState):
        # 在执行任务前获取信号量
        async with semaphore:
            # ainvoke 会异步地完整运行子图并返回最终状态
            return await subgraph.ainvoke(usecase_state)

    # 为每个 usecase 创建一个子图调用任务
    # ainvoke 会异步地完整运行子图并返回最终状态
    tasks = [run_one_subgraph(usecase) for usecase in usecases_to_process]

    # 并发执行所有任务
    results = await asyncio.gather(*tasks)
    
    print(f"--- All {len(results)} subgraphs finished processing. ---")
    # 此节点执行动作，不需要返回状态更新
    return


# Step 2: 定义主图
main_workflow = StateGraph(MainGraphState)
# 添加新的独立节点
main_workflow.add_node("chain_reference_resolver", chain_reference_resolver_node)
main_workflow.add_node("load_and_split", load_and_split_all_usecases_node)
# 添加我们新的并行执行节点
main_workflow.add_node("parallel_executor", run_subgraphs_in_parallel_node)

# Step 3: 链接主图流程
main_workflow.set_entry_point("chain_reference_resolver")
main_workflow.add_edge("chain_reference_resolver", "load_and_split")
main_workflow.add_edge("load_and_split", "parallel_executor")
# 并行执行器完成后，结束整个工作流
main_workflow.add_edge("parallel_executor", END)


# 编译主图
app = main_workflow.compile()


# 用于保存运行结果和配置的函数
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
            "ENABLE_REFERENCE_RESOLUTION": ENABLE_REFERENCE_RESOLUTION,
            "ENABLE_POLICY_SPLITTING": ENABLE_POLICY_SPLITTING,
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
# --- 运行 ---
async def main():
    initial_input = {"usecase_inputs": [{"path": USECASE_JSON_PATH}]}
    # 注意：这里的 config 主要是为 ainvoke 内部的 LangChain 调用服务的，
    # 并发控制现在由 asyncio.gather 掌握。
    # config = {"max_concurrency": MAX_CONCURRENCY} 
    
    # astream 现在用于流式观察主图的状态变化
    async for event in app.astream(initial_input):
        # 打印事件可以帮助观察主图的执行流程
        print(f"Event: {event.keys()}")

    print(f"\n--- Main Workflow Finished ---")

if __name__ == "__main__":
    start_time = time.perf_counter()
    
    # [重要] 确保回调处理器在运行前被重置
    # 这一步现在更加关键，因为它不再在每次调用时自动重置
    token_tracker.reset()

    asyncio.run(main())
    
    elapsed = time.perf_counter() - start_time
    print(f"\nTOTAL EXECUTION TIME: {elapsed:.4f} seconds")

    # 打印最终的token统计信息
    print(f"\n--- Token Usage Summary ---")
    print(f"Total Prompt Tokens: {token_tracker.prompt_tokens}")
    print(f"Total Completion Tokens: {token_tracker.completion_tokens}")
    print(f"Total Tokens: {token_tracker.total_tokens}")


    # 在所有任务结束后，调用保存函数
    save_run_log(LOG_FILE_PATH, elapsed, token_tracker)