"""Provider 抽象基类。

自定义厂商有两种方式：
1. 厂商兼容 OpenAI / Anthropic / Gemini 协议：直接 register_provider(name, base_url=..., api_format=...)。
2. 私有协议：继承 BaseProvider，实现 build_request / parse_response / stream_accumulator，
   再 register_provider(name, provider_class=YourProvider)。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..errors import APIStatusError, StreamError
from ..sse import SSEEvent
from ..types import ChatResponse, StreamChunk


@dataclass
class RequestSpec:
    url: str
    headers: Dict[str, str]
    body: Dict[str, Any]
    query: Dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        # 打印/日志时隐藏鉴权头，避免泄露 api key
        return (f"RequestSpec(url={self.url!r}, headers={_redact(self.headers)!r}, "
                f"query={self.query!r}, body_keys={list(self.body)!r})")


_SENSITIVE_HEADERS = {"authorization", "x-api-key", "api-key", "x-goog-api-key"}


def _redact(headers: Dict[str, str]) -> Dict[str, str]:
    return {k: ("***" if k.lower() in _SENSITIVE_HEADERS else v) for k, v in headers.items()}


class StreamAccumulator:
    """消费 SSE 事件，产出增量 StreamChunk，并在结束时组装完整 ChatResponse。"""

    def __init__(self, provider: "BaseProvider", model: str):
        self.provider = provider
        self.model = model
        self.done = False

    @property
    def completed(self) -> bool:
        """是否收到了完整的结束标记（用于判断流是否被中断）。"""
        return self.done

    def feed(self, event: SSEEvent) -> List[StreamChunk]:
        """处理一个事件；设置 self.done=True 表示可以停止读取。"""
        raise NotImplementedError

    def build(self) -> ChatResponse:
        raise NotImplementedError

    @staticmethod
    def load_json(event: SSEEvent) -> Optional[Dict[str, Any]]:
        data = event.data.strip()
        if not data or data == "[DONE]":
            return None
        try:
            obj = json.loads(data)
        except ValueError:
            return None
        if isinstance(obj, dict) and obj.get("error") and not obj.get("choices") and not obj.get("candidates"):
            err = obj["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise StreamError(f"stream error: {msg}")
        return obj


class BaseProvider:
    api_format: str = "base"
    default_base_url: Optional[str] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        **options: Any,
    ):
        self.name = name or self.api_format
        self.api_key = api_key
        self.base_url = (base_url or self.default_base_url or "").rstrip("/")
        if not self.base_url:
            raise ValueError(f"provider '{self.name}' 需要 base_url")
        self.default_model = default_model
        self.headers = dict(headers or {})
        self.options = options

    # ---- 子类实现 ----------------------------------------------------------
    def build_request(self, messages: List[Dict[str, Any]], model: str, stream: bool, params: Dict[str, Any]) -> RequestSpec:
        raise NotImplementedError

    def parse_response(self, data: Dict[str, Any], model: str) -> ChatResponse:
        raise NotImplementedError

    def stream_accumulator(self, model: str) -> StreamAccumulator:
        raise NotImplementedError

    # ---- 公共工具 ----------------------------------------------------------
    @staticmethod
    def check_error_body(data: Any) -> None:
        """部分厂商出错时返回 HTTP 200 + {"error": ...}。"""
        if isinstance(data, dict) and data.get("error") and not data.get("choices") and not data.get("content"):
            err = data["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise APIStatusError(msg or "unknown error", status_code=200, body=data)

    @staticmethod
    def clean_message(msg: Dict[str, Any]) -> Dict[str, Any]:
        """去掉 "_" 开头的私有字段。"""
        return {k: v for k, v in msg.items() if not k.startswith("_")}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} base_url={self.base_url!r}>"
