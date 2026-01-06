import asyncio
import json
import os
from typing import List, Dict, Any, TypedDict

import aiofiles
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

from langgraph.graph import StateGraph, END

# --- 准备工作：设置你的 OpenAI API Key ---
API_KEY_PATH = r"C:\Users\34085\Desktop\Agent\ALL_API_KEY1.txt"
with open(API_KEY_PATH, "r", encoding='utf-8') as f:
    api_key = f.read().strip()
llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=api_key, base_url="https://4zapi.com/v1/" )

# 1. 输入和输出路径配置
INPUT_FILE_PATH = r"ODRL_V3\data_preparation\merged_final_usecases.json"  # 输入文件路径
OUTPUT_DIR = r"ODRL_V3\data_preparation\E3_usecases"                     # 输出文件夹路径
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "token_usage_log.json")         # Token 消耗日志文件路径

# 2. 并发控制
# 设置同时调用 LLM 的最大任务数，以防止速率限制错误并控制成本
MAX_CONCURRENCY = 70
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


# ----------------------------------------------------------------------------
# 1. 定义图的状态 (Graph State)
# ----------------------------------------------------------------------------
class OrchestratorState(TypedDict):
    """
    定义在 LangGraph 中流转的数据结构。

    Attributes:
        input_filepath: 输入的 use case JSON 文件路径。
        use_cases: 从文件中加载的 use case 字典 (key: usecase_id, value: description)。
        classified_use_cases: 经过编排器分类后的 use case 列表。
        total_tokens: 运行过程中消耗的总 token 数。
        prompt_tokens: 运行过程中消耗的提示 token 数。
        completion_tokens: 运行过程中消耗的补全 token 数。
        successful_requests: 成功的 LLM 请求次数。
    """
    input_filepath: str
    use_cases: Dict[str, str]
    classified_use_cases: List[Dict[str, Any]]
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    successful_requests: int


# ----------------------------------------------------------------------------
# 2. 定义 LLM 的结构化输出模型
# ----------------------------------------------------------------------------
class WorkflowChoice(BaseModel):
    """为 LLM 的决策定义一个清晰的输出结构。"""
    workflow_id: int = Field(..., description="必须是 1, 2, 或 3 中的一个整数。")
    reasoning: str = Field(..., description="做出此选择的简要理由。")


# ----------------------------------------------------------------------------
# 3. LangGraph 的节点实现
# ----------------------------------------------------------------------------
async def load_use_cases_node(state: OrchestratorState) -> OrchestratorState:
    """
    节点 1: 异步加载输入的 JSON 文件。
    [修改] 直接加载整个 JSON 对象作为 use cases 字典。
    """
    print("--- 节点 1: 开始加载 Use Cases ---")
    input_path = state["input_filepath"]
    try:
        async with aiofiles.open(input_path, mode='r', encoding='utf-8') as f:
            content = await f.read()
            # [修改] 直接将加载的 JSON 对象赋值给 use_cases
            data = json.loads(content)
            state["use_cases"] = data
            print(f"成功加载 {len(state['use_cases'])} 个 Use Cases (键值对)。")
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 at {input_path}")
        state["use_cases"] = {}
    except json.JSONDecodeError:
        print(f"错误: 无法解析 JSON from {input_path}")
        state["use_cases"] = {}
    
    # 初始化 token 计数器
    state["total_tokens"] = 0
    state["prompt_tokens"] = 0
    state["completion_tokens"] = 0
    state["successful_requests"] = 0
    
    return state


# ... (替换旧的 orchestrator_node 函数)

async def orchestrator_node(state: OrchestratorState) -> OrchestratorState:
    """
    节点 2: 核心编排器，使用 LLM 异步为每个 use case 分配工作流，并控制并发。
    [已修正]
    """
    print("\n--- 节点 2: Orchestrator 开始决策 ---")
    use_cases = state["use_cases"]
    if not use_cases:
        print("没有 Use Cases 需要处理。")
        state["classified_use_cases"] = []
        return state

    # 初始化 LLM 和 Prompt
    structured_llm = llm.with_structured_output(WorkflowChoice)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
你是一个基于 "Orchestrator-Workers" 结构的多智能体工作流的中央编排器 (Orchestrator)。
你的核心职责是根据输入 use case 的特点，协调并组合不同的 Worker，以形成最高效的工作路径。

--- Worker 定义 ---
*   **重写器 (Rewriter)**: 负责处理包含内部交叉引用的策略。它通过解析依赖关系，将相关条款的内容“内联”到主条款中，从而生成一个单一、扁平化、无内部引用的完整策略文本。
*   **拆分器 (Splitter)**: 负责将一个 use case 分解为多个逻辑上独立的子策略。当规则的核心上下文发生根本性变化时，它会执行拆分。
*   **生成器 (Generator)**: (隐含在所有工作流的最后) 负责根据最终处理好的策略生成 ODRL (Open Digital Rights Language) 策略。

--- 工作路径 (Workflow) 定义 ---
你的任务是从以下三种预设的工作路径中选择最合适的一条：

1.  **工作路径 1 (直接生成)**:
    - **组合**: 直接调用 **生成器**。
    - **描述**: 适用于简单、原子化的策略。这些策略已经足够清晰和具体，不包含任何内部引用，也不需要被拆分，可以直接执行。
    - **选择时机**: 当 use case 是一个“即用型”的、单一且明确的任务时选择此项。对应编号 `1`。

2.  **工作路径 2 (拆分后生成)**:
    - **组合**: **拆分器** -> **生成器**。
    - **描述**: 适用于复杂的策略。这些策略虽然服务于一个整体目标，但由多个**逻辑上可以分离的单元**组成。拆分的目的是将一个复杂任务分解为多个简单任务，以提高生成质量。
    - **选择时机**: 当 use case 不包含内部引用，但符合以下**任一特征**时，**必须**选择此路径：
        *   **1. 包含多个顺序步骤的流程**: 文本描述了一个有先后顺序的工作流（如审核、报告、争议解决流程）。每个步骤（如“申请受理”、“技术验证”、“信用评估”）都应被视为一个独立的单元。
        *   **2. 包含不同主题的条款清单**: 文本列出了一系列关于同一主题（如账户注册、数据使用）但内容各异的条款。每个条款（如“最低年龄”、“账户保密”、“侵权处理”）都应被视为一个独立的单元。
        **【核心原则】**: 你的目标是识别出那些可以通过分解来降低复杂性的用例。对应编号 `2`。
        注意：其实比较简单的usecase也不必要一定要拆分（也就是被拆分的部分和剩余部分的量级差不多，更适合被拆分）
                  
3.  **工作路径 3 (重写+拆分后生成)**:
    - **组合**: **重写器** -> **拆分器** -> **生成器**。
    - **描述**: 适用于最复杂的策略。这些策略既包含需要被解决的内部条款引用，并且在引用被解决后，可能还需要进一步拆分为多个独立的子策略。
    - **选择时机**: 当 use case 中明确存在条款间的引用关系时，必须选择此项。**引用关系可以是显性的（如“违反第三十五条规定...”），也可以是隐性的（如 'Notwithstanding subdivision (a)...'）。任何导致一个条款依赖于另一个条款内容的表述，都应被视为需要重写。** 对应编号 `3`。

---

你必须根据下面的 use case 内容，返回最有效工作路径的编号和选择理由。
"""),
        ("human", "请为以下 use case 分配工作路径:\n\n```json\n{use_case}\n```")
    ])

    chain = prompt_template | structured_llm

    async def classify_use_case(use_case_id: str, use_case_description: str) -> Dict[str, Any]:
        """
        异步调用 LLM 对单个 use case 进行分类，并使用信号量控制并发。
        [修改] 函数接受 id 和 description 两个参数。
        """
        async with semaphore:
            # [修改] 将 id 和 description 组合成一个完整的 use case 对象，再转换为 JSON 字符串。
            # 这样做能为 LLM 提供最完整的上下文。
            full_use_case_obj = {"id": use_case_id, "description": use_case_description}
            use_case_str = json.dumps(full_use_case_obj, ensure_ascii=False, indent=2)
            
            # 使用 get_openai_callback 追踪 token 消耗
            with get_openai_callback() as cb:
                choice = await chain.ainvoke({"use_case": use_case_str})
                
                print(f"Use Case (ID: {use_case_id}) -> 分配到工作流 {choice.workflow_id} (理由: {choice.reasoning}) | Tokens: {cb.total_tokens}")
                
                # [修改] 返回的结构包含原始的 id, description, workflow_id 以及本次调用的 token 信息
                return {
                    "id": use_case_id,
                    "description": use_case_description,
                    "workflow_id": choice.workflow_id,
                    "tokens": {
                        "total": cb.total_tokens,
                        "prompt": cb.prompt_tokens,
                        "completion": cb.completion_tokens,
                        "requests": cb.successful_requests
                    }
                }

    # [修复] 迭代字典的 items()，将 key 和 value 分别作为参数传递给 classify_use_case。
    # 这修复了原始的 TypeError。
    tasks = [classify_use_case(key, value) for key, value in use_cases.items()]
    classified_results = await asyncio.gather(*tasks)

    # 聚合所有任务的 token 消耗
    total_tokens = sum(r["tokens"]["total"] for r in classified_results)
    prompt_tokens = sum(r["tokens"]["prompt"] for r in classified_results)
    completion_tokens = sum(r["tokens"]["completion"] for r in classified_results)
    successful_requests = sum(r["tokens"]["requests"] for r in classified_results)

    state["classified_use_cases"] = classified_results
    state["total_tokens"] = total_tokens
    state["prompt_tokens"] = prompt_tokens
    state["completion_tokens"] = completion_tokens
    state["successful_requests"] = successful_requests
    
    print(f"\n--- Orchestrator 决策完成 (并发数: {MAX_CONCURRENCY}) ---")
    print(f"--- Token 统计: 总计 {total_tokens} | 提示 {prompt_tokens} | 补全 {completion_tokens} ---")
    return state


# ... (替换旧的 save_results_node)

async def save_results_node(state: OrchestratorState) -> OrchestratorState:
    """
    节点 3: 将分类结果异步写入到三个不同的 JSON 文件中。
    [已修正]
    """
    print("\n--- 节点 3: 开始分组并保存结果 ---")
    classified_use_cases = state.get("classified_use_cases", [])

    # 初始化分组结果为空字典
    grouped_results = {1: {}, 2: {}, 3: {}}
    for item in classified_use_cases:
        # [修复] 根据 orchestrator_node 返回的新结构来正确解析数据。
        # 原有代码会因为找不到 "key" 和 "value" 而报错。
        usecase_id = item["id"]
        description = item["description"]
        workflow_id = item["workflow_id"]
        
        if workflow_id in grouped_results:
            # 将原始的键值对添加到对应的分组字典中
            grouped_results[workflow_id][usecase_id] = description

    # 使用全局变量
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_files = {
        1: os.path.join(OUTPUT_DIR, "workflow_1_direct_run.json"),
        2: os.path.join(OUTPUT_DIR, "workflow_2_split_only.json"),
        3: os.path.join(OUTPUT_DIR, "workflow_3_rewrite_split.json")
    }

    async def write_file(workflow_id: int, data: Dict[str, Any]):
        """异步写入单个 JSON 文件。"""
        filepath = output_files[workflow_id]
        async with aiofiles.open(filepath, mode='w', encoding='utf-8') as f:
            await f.write(json.dumps(data, indent=4, ensure_ascii=False))
        print(f"已将 {len(data)} 个 use cases 写入到: {filepath}")

    # 使用 asyncio.gather 并发写入所有文件，只写入非空的分组
    write_tasks = [write_file(wf_id, ucs) for wf_id, ucs in grouped_results.items() if ucs]
    await asyncio.gather(*write_tasks)

    print("--- 所有结果已保存 ---")
    return state

async def log_statistics_node(state: OrchestratorState) -> OrchestratorState:
    """
    节点 4: 将运行统计数据写入日志文件。
    """
    print("\n--- 节点 4: 开始记录统计数据 ---")
    
    stats = {
        "model_used": llm.model_name,
        "total_use_cases_processed": state.get("successful_requests", 0),
        "max_concurrency": MAX_CONCURRENCY,
        "token_usage": {
            "total_tokens": state.get("total_tokens", 0),
            "prompt_tokens": state.get("prompt_tokens", 0),
            "completion_tokens": state.get("completion_tokens", 0),
        }
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    async with aiofiles.open(LOG_FILE_PATH, mode='w', encoding='utf-8') as f:
        await f.write(json.dumps(stats, indent=4, ensure_ascii=False))
        
    print(f"--- 统计数据已写入到: {LOG_FILE_PATH} ---")
    return state

# ----------------------------------------------------------------------------
# 4. 构建并编译 LangGraph
# ----------------------------------------------------------------------------

# 实例化 StateGraph
workflow = StateGraph(OrchestratorState)

# 添加节点
workflow.add_node("loader", load_use_cases_node)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("saver", save_results_node)
workflow.add_node("logger", log_statistics_node) # 新增日志节点

# 设置图的流转路径
workflow.set_entry_point("loader")
workflow.add_edge("loader", "orchestrator")
workflow.add_edge("orchestrator", "saver")
workflow.add_edge("saver", "logger") # saver 之后调用 logger
workflow.add_edge("logger", END)     # logger 之后结束

# 编译图
app = workflow.compile()

async def main():
    """主执行函数"""
    
    inputs = {"input_filepath": INPUT_FILE_PATH}
    
    print("\n===================================")
    print("  Orchestrator LangGraph 开始运行  ")
    print("===================================\n")
    
    # 异步执行图
    # ainvoke 会流式返回每个节点执行后的状态，这里我们只关心最终结果
    # 如果需要查看中间步骤，可以迭代 `app.astream(inputs)`
    final_state = await app.ainvoke(inputs)
    
    print("\n===================================")
    print("  Orchestrator LangGraph 运行结束  ")
    print("===================================\n")


if __name__ == "__main__":

    asyncio.run(main())