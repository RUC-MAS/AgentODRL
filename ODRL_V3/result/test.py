from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# --- 1. 配置您的模型信息 ---
# ⚠️ 重要：请将这里的 URL 替换为您从 AutoDL 获取的真实公-网地址
# 并且确保地址末尾有 "/v1"
# API_BASE_URL = "https://YOUR_AUTODL_PUBLIC_URL:PORT/v1"
API_BASE_URL = "http://localhost:8000/v1" # 隧道建立后，您的 API 不再是通过公网地址访问，而是通过您本地电脑的端口访问。


# 这个路径必须和您启动 vLLM 服务时 --model 参数的值完全一样
MODEL_PATH = "/root/autodl-tmp/sft-qwen3-merged/"


# --- 2. 初始化 LangChain 的 ChatOpenAI 模型 ---
# 我们使用 ChatOpenAI 类，因为它与 vLLM 提供的 OpenAI 兼容接口完全匹配
llm = ChatOpenAI(
    # 指向您的 vLLM 服务器
    openai_api_base=API_BASE_URL,
    
    # 指定要使用的模型，即您微调好的模型
    model=MODEL_PATH,
    
    # vLLM 不需要 API Key，但 LangChain 的这个字段是必需的，所以我们填入一个任意的字符串
    openai_api_key="not-needed",
    
    # 其他可选参数，例如温度
    temperature=0.2,
    # 让模型的返回不包含thinking模式的内容
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)


# --- 3. 调用模型并获取回复 ---
print("正在向您的模型发送请求...")

# 构建对话消息
messages = [
    ("system","从给定的规则文本中提取所有核心语义点。“对于接下来的任务，我希望你直接给出最终答案。请省略你的思考过程，不要输出<think>标签。”"),
   ("user","制造业员工可随时基于其个人情况反对因“为完成生产安全任务或行使工厂管理权之必要”或“为工厂或第三方实现合法利益而必要”而处理与其相关的工作记录数据，包括基于这些规定的绩效分析。此项规定不适用于政府机关履行监管职责时的数据处理。除非工厂方能证明其处理理由高于员工的利益、权利和自由，或为提出、行使或辩护合法申诉，否则不得继续处理该等工作记录数据。"),
]

# 调用模型
response = llm.invoke(messages)

# 打印模型的回复内容
print("\n模型的回复：")
print(response.content)