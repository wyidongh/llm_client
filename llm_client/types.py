"""数据结构：Usage / ToolCall / ChatResponse / StreamChunk。"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Usage:
    """一次请求（或累计）的 token 统计。

    prompt_tokens      请求（输入）token 数，包含命中缓存的部分
    completion_tokens  响应（输出）token 数，包含推理/思考 token
    total_tokens       合计
    cached_tokens      输入中命中缓存的 token 数（厂商支持时）
    reasoning_tokens   输出中推理/思考 token 数（厂商支持时）
    estimated          True 表示厂商未返回 usage，数值为本地估算
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    estimated: bool = False

    def __post_init__(self):
        if not self.total_tokens:
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            estimated=self.estimated or other.estimated,
        )

    @property
    def is_empty(self) -> bool:
        return not (self.prompt_tokens or self.completion_tokens or self.total_tokens)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        s = f"输入={self.prompt_tokens} 输出={self.completion_tokens} 合计={self.total_tokens}"
        if self.cached_tokens:
            s += f" 缓存命中={self.cached_tokens}"
        if self.reasoning_tokens:
            s += f" 推理={self.reasoning_tokens}"
        if self.estimated:
            s += " (估算)"
        return s


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = ""  # JSON 字符串

    def parsed_arguments(self) -> Any:
        return json.loads(self.arguments) if self.arguments else {}

    def to_openai(self) -> Dict[str, Any]:
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class ChatResponse:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None  # stop / length / tool_calls / content_filter / refusal / ...
    usage: Usage = field(default_factory=Usage)
    model: Optional[str] = None
    provider: Optional[str] = None
    id: Optional[str] = None
    latency: Optional[float] = None  # 总耗时（秒）
    first_token_latency: Optional[float] = None  # 流式首 token 耗时（秒）
    interrupted: bool = False  # 流式被中途关闭/断开
    raw: Any = None  # 非流式：原始 JSON
    # 厂商私有的回放数据（如 Anthropic 的 thinking 签名块），写入历史消息的 "_xxx" 字段
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> Dict[str, Any]:
        """转换为可放入对话历史的 assistant 消息（OpenAI 风格）。"""
        msg: Dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
            if not self.content:
                msg["content"] = None
        if self.reasoning_content:
            msg["_reasoning"] = self.reasoning_content
        msg.update(self.extras)
        return msg

    def __str__(self) -> str:
        return self.content


@dataclass
class StreamChunk:
    content: str = ""
    reasoning_content: str = ""
    tool_call: Optional[ToolCall] = None  # 一个工具调用完整解析后才产出
    finish_reason: Optional[str] = None


# ---------------------------------------------------------------- token 估算

_CJK_RE = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿가-힯＀-￯]")


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（仅当厂商不返回 usage 时兜底使用）。

    中日韩字符约 1 token/字，其他字符约 4 字符/token。
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    return cjk + max(0, len(text) - cjk + 3) // 4


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        total += 4  # 每条消息的格式开销
        content = m.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += estimate_tokens(part["text"])
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            total += estimate_tokens(fn.get("name", "")) + estimate_tokens(fn.get("arguments", ""))
    return total
