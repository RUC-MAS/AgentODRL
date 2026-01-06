# data_augmentation_only_workflow.py

import asyncio
import json
import operator
from typing import List, TypedDict, Annotated

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import Send

# --- 1. 配置 ---
# 输入文件：包含待扩充的 usecase
INPUT_JSON_PATH = r"ODRL_V3\data_preparation\complex_usecases_progression.json"
# 输出文件：保存扩充后的 usecase
OUTPUT_JSON_PATH = r"ODRL_V3\result\generated_usecases\final_usecases\gen_complex_usecases_progression3.json"
# OpenAI API Key 路径
API_KEY_PATH = r"C:\Users\34085\Desktop\Agent\ALL_API_KEY.txt"
# 使用的 LLM 模型
LLM_MODEL_NAME = "gpt-4.1"
# 每个 usecase 生成的变体数量
NUM_VARIATIONS_PER_USECASE = 5
# 并发数量
MAX_CONCURRENCY = 41

# --- 2. Pydantic 模型 & LangGraph 状态定义 ---

class JsonStringList(BaseModel):
    """用于LLM结构化输出的Pydantic模型"""
    items: List[str] = Field(description="A list of generated textual variations of the data rule.")

class WorkflowState(TypedDict):
    """定义工作流的全局状态 (已简化)"""
    usecases_to_augment: List[str]
    augmented_texts: Annotated[list, operator.add]

# --- 3. LLM 及 Prompt 初始化 (只保留扩充部分) ---
try:
    with open(API_KEY_PATH, "r", encoding='utf-8') as f:
        api_key = f.read().strip()
except FileNotFoundError:
    print(f"错误: 未找到 API 密钥文件于 {API_KEY_PATH}")
    exit()

llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.2, api_key=api_key , base_url="https://zzzzapi.com/v1")
structured_llm = llm.with_structured_output(JsonStringList)

# 变体生成 Prompt ([已再次修改] 强化逻辑自洽和数量约束)
augmentation_prompt = ChatPromptTemplate.from_template(
    """As a Data Space Governance Specialist, your mission is to create **exactly {num_variations}** semantically equivalent variations of a data usage policy.
    Your goal is to generate diverse, logically coherent examples that reflect real-world, multi-party data sharing scenarios within a governed data space.
    Preserve the core ODRL logic while creatively altering the contextual details.

**1. Core Preservation (MUST RETAIN):**
- Rule type: Permission/Prohibition/Obligation
- Action-Target binding (e.g., 'access'→'patient data')
- Constraint categories (temporal, purpose, location, event, quantity)
- Logical operators (must, may, shall not, is required to)
- Negation scope (if present, e.g., "except PII data")

**2. Context Transformation (MUST ALTER):**
| Element       | Transformation Patterns (Data Space Focus)                                                              |
|---------------|---------------------------------------------------------------------------------------------------------|
| **Participants** | Shift participant roles and industry (e.g., 'Mobility Provider' → 'Smart City Authority' or 'Automotive OEM' → 'Tier-1 Supplier') |
| **Data Assets** | Change Data Asset domain (e.g., 'Vehicle Telemetry Data' → 'Aggregated Traffic Flow Data' or 'Patient Cohort Data' → 'Clinical Trial Results') |
| **Locations** | Alter geographic or logical scope (e.g., 'EU Data Space' → 'Catena-X Network' or 'Gaia-X Trusted Zone') |
| **Timeframes** | Modify duration/endpoints (fixed date ↔ relative period, e.g., 'until 2026' ↔ 'for the project duration') |
| **Quantifiers** | Vary numerical limits (e.g., '1000 records/day' ↔ 'up to 10GB total' or '3 API calls/hour')           |
| **Use Cases** | Replace Use Case descriptors (e.g., 'Supply Chain Optimization' → 'Carbon Footprint Reporting' or 'Federated AI Training') |

**3. Semantic Equivalence Validation:**
Each variation must pass:
✓ Identical permission/prohibition structure.
✓ Equivalent constraint types and relationships.
✓ Unchanged action-target binding strength.
✓ Transformed context must still represent a valid, plausible data space interaction.
✓ **Logical Coherence**: The combination of Participant, Data Asset, and Use Case must be plausible and reflect a realistic scenario. (e.g., An 'Automotive OEM' would use 'Component Lifecycle Data' for 'Circular Economy Analysis', NOT for 'Clinical Trial Results').

**4. Master Example (Data Space Context):**
› **Original:** "The 'Smart Mobility Data Space' allows 'CityBus Operator' to use 'Real-time GPS Traces' for the 'Route Optimization' use case until 2026."
› **Variation:** "The 'Catena-X Automotive Network' permits 'Global Auto Parts Inc.' to access 'Component Lifecycle Data' for 'Circular Economy Analysis' through 2027."
  ✓ **Validation:** Permission structure, participant roles, data asset, use case, and temporal constraints are all preserved and transformed within a data space paradigm.

**5. Output Protocol (STRICT):**
① **CRITICAL: Generate EXACTLY {num_variations} variations.** The final JSON output must contain this precise number of items. Do not generate more or fewer. This is a primary instruction.
② Each variation must be a complete, standalone, and logically coherent rule statement.
③ Output ONLY a valid JSON object in the format: {{"items": ["var1", "var2", ...]}}

**Original Rule:** "{use_case_text}"
"""
)

# --- 4. 节点与边的函数定义 (已简化) ---

def load_usecases_from_json_node(state: WorkflowState) -> dict:
    """节点 1: 从本地 JSON 文件加载 usecase 文本。
    [已修改] 此节点现在从 JSON 对象的 *值* 中提取 usecases，忽略键。
    """
    print(f"--- [1. Loader] 正在从 {INPUT_JSON_PATH} 加载规则... ---")
    usecases = []  # 默认值
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查加载的数据是否为字典 (JSON 对象)
        if not isinstance(data, dict):
            print(f"错误: JSON 文件 '{INPUT_JSON_PATH}' 的内容不是一个有效的 JSON 对象 (key-value 字典)。")
        else:
            # 从字典的值中提取所有的 usecase 字符串
            usecases = list(data.values())
            if not usecases:
                print(f"警告: 从 JSON 文件 '{INPUT_JSON_PATH}' 中未能提取到任何 usecase 值。")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到于 '{INPUT_JSON_PATH}'")
    except json.JSONDecodeError:
        print(f"错误: 无法解析 JSON 文件 '{INPUT_JSON_PATH}'")
    
    if usecases:
        print(f"--- [1. Loader] 加载完成，共 {len(usecases)} 条规则待处理。")
    else:
        print("--- [1. Loader] 未加载到任何规则，工作流将终止。")

    # 无论成功与否，都返回一个符合 WorkflowState 结构的字典
    return {
        "usecases_to_augment": usecases,
        "augmented_texts": [],
    }

def dispatcher_for_augmentation(state: WorkflowState) -> List[Send]:
    """边 A: 为每条原始 usecase 分发一个增广任务。"""
    print(f"--- [Dispatcher] 正在为 {len(state['usecases_to_augment'])} 条规则分发增广任务... ---")
    return [Send("augment_node", text) for text in state["usecases_to_augment"]]

async def augment_node(use_case_text: str) -> dict:
    """节点 2 (并行): 接收一条 usecase，生成变体。"""
    print(f"      [Augment] 正在处理: '{use_case_text[:60]}...'")
    chain = augmentation_prompt | structured_llm
    response = await chain.ainvoke({
        "num_variations": NUM_VARIATIONS_PER_USECASE,
        "use_case_text": use_case_text
    })
    # 只返回新生成的变体
    return {"augmented_texts": [response.items]}

def save_augmented_usecases_node(state: WorkflowState) -> dict:
    """节点 3 (最终): 聚合所有结果并按指定格式保存到 JSON 文件。"""
    print("--- [Saver] 所有增广任务完成。正在整理并保存结果... ---")
    
    # 将 state 中收集到的多层列表展平为单层列表
    all_new_usecases = [text for text_list in state["augmented_texts"] for text in text_list]
    
    # 构建您指定的 key-value 格式
    output_data = {
        f"usecase_{i+1}": text
        for i, text in enumerate(all_new_usecases)
    }
    
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        # 使用 indent=4 使 JSON 文件格式优美，易于阅读
        json.dump(output_data, f, ensure_ascii=False, indent=4)
            
    print(f"--- [Saver] 成功生成 {len(all_new_usecases)} 条新 usecase，并保存至: {OUTPUT_JSON_PATH} ---")
    return {}

# --- 5. 构建图 (已简化) ---

graph_builder = StateGraph(WorkflowState)

# 注册所有节点
graph_builder.add_node("load_usecases", load_usecases_from_json_node)
graph_builder.add_node("augment_node", augment_node)
graph_builder.add_node("save_results", save_augmented_usecases_node)

# 设置入口点
graph_builder.set_entry_point("load_usecases")

# 定义图的边
# 1. 加载数据后，通过 dispatcher 将任务分发给 augment_node
graph_builder.add_conditional_edges("load_usecases", dispatcher_for_augmentation)
# 2. 所有 augment_node 任务完成后，自动聚合结果，然后进入 save_results 节点
graph_builder.add_edge("augment_node", "save_results")
# 3. 保存后，结束工作流
graph_builder.add_edge("save_results", END)

# 编译最终的图
graph = graph_builder.compile()

# --- 6. 运行工作流 ---

async def main():
    """主函数，配置并运行 LangGraph 工作流"""
    print("--- 工作流启动 ---")
    config = {"max_concurrency": MAX_CONCURRENCY}
    
    async for event in graph.astream({}, config=config):
        # 打印事件以监控进度
        print(f"--- Event: {list(event.keys())[0]} ---")

    print("\n--- 工作流执行完毕！---")

if __name__ == "__main__":
    asyncio.run(main())