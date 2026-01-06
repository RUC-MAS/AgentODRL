# data_generation_workflow.py

import asyncio
import json
import operator
from typing import List, TypedDict, Annotated

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from mongodb import MongoDBManager

# --- 1. 配置 (请根据您的实际情况替换) ---
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "odrl3"
USECASE_COLLECTION_NAME = "cu_progression_4_1_nano"
LLM_MODEL_NAME = "gpt-4.1"
NUM_VARIATIONS_PER_USECASE = 5
OUTPUT_FILE_PATH = r"ODRL_V3\result\data_for_sft\sft_data.jsonl"

# --- 2. Pydantic 模型 & LangGraph 状态定义 ---

class FineTuneData(BaseModel):
    text: str
    semantic_points: List[str]

class JsonStringList(BaseModel):
    items: List[str]

class WorkflowState(TypedDict):
    """定义工作流的全局状态"""
    usecases_to_augment: List[str]
    augmented_texts: Annotated[list, operator.add]
    final_data: Annotated[list, operator.add]

# --- 3. LLM 及 Prompt 初始化 ---
API_KEY_PATH = r"C:\Users\34085\Desktop\Agent\OPENAI_API_KEY.txt"
with open(API_KEY_PATH, "r", encoding='utf-8') as f:
    api_key = f.read().strip()
llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.3, api_key=api_key)
structured_llm = llm.with_structured_output(JsonStringList)

augmentation_prompt = ChatPromptTemplate.from_template(
    """As a Data Rule Transformation Specialist, create {num_variations} semantically equivalent variations of this data policy. 
    Preserve the core ODRL logic while creatively altering contextual details.

**1. Core Preservation (MUST RETAIN):**
- Rule type: Permission/Prohibition/Obligation
- Action-Target binding (e.g., 'access'→'patient data')
- Constraint categories (temporal, purpose, location, event, quantity)
- Logical operators (must, may, shall not, is required to)
- Negation scope (if present, e.g., "except PII data")

**2. Context Transformation (MUST ALTER):**
| Element        | Transformation Patterns                                       |
|----------------|---------------------------------------------------------------|
| **Entities** | Shift industry context (healthcare→finance→manufacturing)     |
| **Data Assets**| Change dataset domain (medical→industrial→geospatial)         |
| **Locations** | Alter geographic scope (EU→APAC→NAFTA regions)                |
| **Timeframes** | Modify duration/endpoints (fixed date↔relative period)       |
| **Quantifiers**| Vary numerical limits (count↔frequency↔volume)               |
| **Projects** | Replace purpose descriptors (keep category: research↔analytics) |

**3. Semantic Equivalence Validation:**
Each variation must pass:
✓ Identical permission/prohibition structure
✓ Equivalent constraint types and relationships
✓ Unchanged action-target binding strength
✓ Preserved negation scope (if applicable)

**4. Master Example:**
› **Original:** "PharmaCorp Inc. grants access to Patient Data for Cancer Research until 2026 except anonymized records"
› **Variation:** "AgriGen SA permits use of Crop Data for Sustainability through 2027 excluding GMO samples"
  ✓ **Validation:** Permission structure + purpose + temporal + exclusion all preserved

**5. Output Protocol:**
① Generate exactly {num_variations} variations
② Each as complete, standalone rule statement
③ Output ONLY valid JSON: ["var1", "var2", ...]

**Original Rule:** "{use_case_text}"
"""
)

extraction_prompt = ChatPromptTemplate.from_template(
    """As a Semantic Unit Extractor, identify all atomic information units in this data policy. 
    Extract EXACT text spans defining the rule's core semantics, following this strict protocol:

**1. Extraction Protocol:**
- **Scan Anchors**:
  • Granting/Receiving entities (preserve full naming)
  • Action verbs with modifiers and negations (e.g., "securely process", "must not share")
  • Data asset references (include descriptors)
  • Permission/prohibition markers (e.g., "grants", "prohibits")
  • Constraint clusters (temporal/purpose/location/quantity)
  • Obligation phrases (e.g., "must report", "is required to")

- **Extraction Rules**:
  ✓ **Fidelity**: Preserve exact wording, order and capitalization
  ✓ **Completeness**: Capture entire phrases (e.g., full dates "2025-12-31")
  ✓ **Compound Units**: Keep logical units intact ("must not process externally")
  ✗ **Exclusion**: Omit standalone syntactic words (that, which, etc.)

**2. Quality Assurance:**
Before output, verify:
- Every semantic unit is extracted
- No phrase truncation (full date/time/names)
- Compound units preserved as single spans
- No isolated syntactic words included

**3. Validation Examples:**
✓ **Valid Extraction**  
  Text: "The Data Hub grants the Urban Planning Dept access to Traffic Data until 2025-12-31"  
  Output: `["The Data Hub", "grants", "the Urban Planning Dept", "access", "Traffic Data", "until 2025-12-31"]`

✗ **Invalid Extraction**  
  Text: "Analytics teams may process sales records except PII data"  
  Error: `["Analytics", "may process", "sales records", "except", "PII data"]`  
  Fix: `["Analytics teams", "may process", "sales records", "except PII data"]`  
  (Reason: "except PII data" is a single constraint unit)

**4. Output Format:** STRICTLY JSON array: `["unit1", "unit2", ...]`

**Extract from:** "{text}"
"""
)

# --- 4. 节点与边的函数定义 ---

# --- 4a. MongoDB Projection ---
# 定义投影，确保只从数据库获取我们需要的字段
MONGO_PROJECTION = {"policies.text": 1, "_id": 0}

async def fetch_usecases_node(state: WorkflowState) -> dict:
    """节点 1: 从 MongoDB 精确获取 usecase 文本，并初始化状态。"""
    print("--- [1. Fetcher] 正在从 MongoDB 加载规则... ---")
    mongo_manager = MongoDBManager(mongo_uri=MONGO_URI, mongo_db_name=MONGO_DB_NAME)
    documents = await mongo_manager.fetch_all_rules(
        USECASE_COLLECTION_NAME,
        projection=MONGO_PROJECTION
    )
    await mongo_manager.close_connection()

    usecases = []
    for doc in documents:
        if "policies" in doc and isinstance(doc["policies"], list):
            for policy in doc["policies"]:
                # 由于投影，policy字典中理论上只有'text'字段
                if "text" in policy and policy["text"]:
                    usecases.append(policy["text"])
    
    print(f"--- [1. Fetcher] 加载完成，共 {len(usecases)} 条规则待处理。")
    return {
        "usecases_to_augment": usecases,
        "augmented_texts": [],
        "final_data": [],
    }

def dispatcher_for_augmentation(state: WorkflowState) -> List[Send]:
    """边 A: 读取所有原始 usecase，为每条规则分发一个增广任务。"""
    print(f"--- [Dispatcher A] 正在为 {len(state['usecases_to_augment'])} 条规则分发增广任务... ---")
    return [Send("augment_node", text) for text in state["usecases_to_augment"]]

async def augment_node(use_case_text: str) -> dict:
    """节点 2 (并行): 接收一条 usecase，生成变体。"""
    print(f"    [Augment] 正在处理: '{use_case_text[:40]}...'")
    chain = augmentation_prompt | structured_llm
    response = await chain.ainvoke({
        "num_variations": NUM_VARIATIONS_PER_USECASE,
        "use_case_text": use_case_text
    })
    # all_texts_for_this_task = [use_case_text] + response.items
    all_texts_for_this_task = response.items # [use_case_text] 会把最初那条也一并返回
    return {"augmented_texts": [all_texts_for_this_task]}

def dispatcher_for_extraction(state: WorkflowState) -> List[Send]:
    """边 B (同步点): 在所有增广任务完成后，分发提取任务。"""
    all_texts_flat = [text for text_list in state["augmented_texts"] for text in text_list]
    print(f"--- [Dispatcher B] 增广完成。正在为 {len(all_texts_flat)} 段文本分发提取任务... ---")
    return [Send("extract_points_node", text) for text in all_texts_flat]

async def extract_points_node(text: str) -> dict:
    """节点 3 (并行): 接收一段文本，提取语义点。"""
    print(f"    [Extract] 正在提取: '{text[:40]}...'")
    chain = extraction_prompt | structured_llm
    response = await chain.ainvoke({"text": text})
    processed_item = FineTuneData(text=text, semantic_points=response.items)
    return {"final_data": [processed_item.model_dump()]}

def save_results_node(state: WorkflowState) -> dict:
    """节点 4: 聚合所有结果并转换为指令微调格式，然后保存到文件。"""
    all_data = state["final_data"]
    print(f"--- [4. Saver] 所有任务完成。正在将 {len(all_data)} 条数据转换为微调格式并写入文件... ---")
    
    # 定义一个固定的指令，用于所有的数据样本
    instruction_prompt = "从给定的规则文本中提取所有核心语义点。"
    
    with open(OUTPUT_FILE_PATH, 'a', encoding='utf-8') as f: # 以追加模式打开文件
        for item in all_data:
            # 将原始数据转换为 "instruction-input-output" 格式
            fine_tune_record = {
                "instruction": instruction_prompt,
                "input": item["text"],
                "output": json.dumps(item["semantic_points"], ensure_ascii=False) 
            }
            f.write(json.dumps(fine_tune_record, ensure_ascii=False) + '\n')
            
    print(f"--- [4. Saver] 成功保存至: {OUTPUT_FILE_PATH} ---")
    return {}

# --- 5. 构建图 ---

graph_builder = StateGraph(WorkflowState)

graph_builder.add_node("fetch_usecases", fetch_usecases_node)
graph_builder.add_node("augment_node", augment_node)
graph_builder.add_node("extract_points_node", extract_points_node)
graph_builder.add_node("save_results", save_results_node)

graph_builder.set_entry_point("fetch_usecases")
graph_builder.add_conditional_edges("fetch_usecases", dispatcher_for_augmentation)
graph_builder.add_conditional_edges("augment_node", dispatcher_for_extraction)
graph_builder.add_edge("extract_points_node", "save_results")
graph_builder.add_edge("save_results", END)

graph = graph_builder.compile()

# --- 6. 运行工作流 (使用事件流以监控进度) ---

async def main():
    print("--- 工作流启动 ---")
    # config = {"recursion_limit": 200, "max_concurrency": MAX_CONCURRENCY}
    
    async for event in graph.astream({}):
        print(event)

    print("\n--- 工作流执行完毕！---")

if __name__ == "__main__":
    asyncio.run(main())