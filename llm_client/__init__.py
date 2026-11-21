"""llm_client —— 基于 requests 的多厂商大模型 API 客户端。

    from llm_client import LLMClient

    client = LLMClient("deepseek")                     # 读取 DEEPSEEK_API_KEY
    print(client.ask("你好"))

    for text in client.stream("写一首诗").text_stream:  # 流式
        print(text, end="", flush=True)

    s = client.new_session(system="你是助手")           # 多轮会话
    s.chat("我叫小明"); print(s.chat("我叫什么？"))
    print(s.last_usage, s.total_usage)                 # token 统计
"""
from .client import LLMClient, normalize_messages
from .errors import (APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, BadRequestError,
                     InternalServerError, LLMError, NotFoundError, PermissionDeniedError, RateLimitError,
                     StreamError)
from .providers import (AnthropicProvider, AzureOpenAIProvider, BaseProvider, GeminiProvider, OpenAICompatProvider,
                        RequestSpec, StreamAccumulator, create_provider, list_providers, register_format,
                        register_provider)
from .session import ChatSession, TurnRecord
from .stream import StreamResponse
from .types import ChatResponse, StreamChunk, ToolCall, Usage, estimate_tokens

__version__ = "0.1.0"

__all__ = [
    "LLMClient", "ChatSession", "TurnRecord", "StreamResponse", "normalize_messages",
    "ChatResponse", "StreamChunk", "ToolCall", "Usage", "estimate_tokens",
    "BaseProvider", "RequestSpec", "StreamAccumulator",
    "OpenAICompatProvider", "AzureOpenAIProvider", "AnthropicProvider", "GeminiProvider",
    "create_provider", "register_provider", "register_format", "list_providers",
    "LLMError", "APIConnectionError", "APITimeoutError", "APIStatusError", "BadRequestError",
    "AuthenticationError", "PermissionDeniedError", "NotFoundError", "RateLimitError", "InternalServerError",
    "StreamError",
]
