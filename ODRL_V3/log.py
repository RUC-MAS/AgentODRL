# log.py

import asyncio
from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any

class TokenUsageCallbackHandler(AsyncCallbackHandler):
    """一个简化的回调处理器，仅在被调用时累加token."""

    def __init__(self):
        super().__init__()
        self._lock = asyncio.Lock()
        # 将token计数器直接作为实例属性
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def __enter__(self):
        """允许作为上下文管理器使用，在进入时重置计数器。"""
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器。"""
        pass

    def reset(self):
        """重置所有token计数器。"""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """在LLM调用结束时，无条件地累加token。"""
        token_usage = response.llm_output.get("token_usage", {})
        
        # 安全地获取token计数
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)
        
        async with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            # (可选) 移除这里的print，因为主循环会打印最终结果
            # print(f"[Token Tracker] Raw Tokens: {token_usage}. Current Total: {self.total_tokens}")