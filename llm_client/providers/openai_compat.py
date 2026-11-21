"""OpenAI Chat Completions 协议（OpenAI、DeepSeek、通义千问、Kimi、智谱、豆包、硅基流动、Ollama、vLLM 等）。"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..sse import SSEEvent
from ..types import ChatResponse, StreamChunk, ToolCall, Usage
from .base import BaseProvider, RequestSpec, StreamAccumulator


def parse_openai_usage(u: Optional[Dict[str, Any]]) -> Usage:
    if not u:
        return Usage()
    prompt = u.get("prompt_tokens") or 0
    completion = u.get("completion_tokens") or 0
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or u.get("prompt_cache_hit_tokens") or 0
    reasoning = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=u.get("total_tokens") or prompt + completion,
        cached_tokens=cached,
        reasoning_tokens=reasoning,
    )


def _reasoning_of(obj: Dict[str, Any]) -> str:
    # DeepSeek / 千问 / Kimi: reasoning_content；OpenRouter / Ollama: reasoning
    r = obj.get("reasoning_content") or obj.get("reasoning") or ""
    return r if isinstance(r, str) else ""


class OpenAICompatProvider(BaseProvider):
    api_format = "openai"
    default_base_url = "https://api.openai.com/v1"

    def __init__(
        self,
        *args: Any,
        chat_path: str = "/chat/completions",
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer ",
        stream_usage: bool = True,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.chat_path = chat_path
        self.auth_header = auth_header
        self.auth_prefix = auth_prefix
        # 流式时发送 stream_options.include_usage 以获得 token 统计；
        # 若厂商不支持会返回 400，客户端会自动关闭此项重试。
        self.stream_usage = stream_usage

    def endpoint(self, model: str) -> str:
        return self.base_url + self.chat_path

    def build_request(self, messages, model, stream, params) -> RequestSpec:
        params = dict(params)
        effort = params.pop("reasoning_effort", None)
        body: Dict[str, Any] = {"model": model, "messages": [self.clean_message(m) for m in messages]}
        if effort is not None:
            body["reasoning_effort"] = effort
        body.update(params)
        if stream:
            body["stream"] = True
            if self.stream_usage:
                body.setdefault("stream_options", {"include_usage": True})
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"}
        headers.update(self.headers)
        if self.api_key:
            headers[self.auth_header] = self.auth_prefix + self.api_key
        return RequestSpec(self.endpoint(model), headers, body)

    def parse_response(self, data, model) -> ChatResponse:
        self.check_error_body(data)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        tool_calls = [
            ToolCall(
                id=tc.get("id") or f"call_{i}",
                name=(tc.get("function") or {}).get("name", ""),
                arguments=(tc.get("function") or {}).get("arguments") or "",
            )
            for i, tc in enumerate(msg.get("tool_calls") or [])
        ]
        return ChatResponse(
            content=msg.get("content") or "",
            reasoning_content=_reasoning_of(msg),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            usage=parse_openai_usage(data.get("usage")),
            model=data.get("model") or model,
            provider=self.name,
            id=data.get("id"),
            raw=data,
        )

    def stream_accumulator(self, model) -> StreamAccumulator:
        return OpenAIStreamAccumulator(self, model)


class OpenAIStreamAccumulator(StreamAccumulator):
    def __init__(self, provider, model):
        super().__init__(provider, model)
        self.content: List[str] = []
        self.reasoning: List[str] = []
        self.tools: Dict[int, Dict[str, str]] = {}
        self.finish_reason: Optional[str] = None
        self.usage = Usage()
        self.id: Optional[str] = None
        self.resp_model: Optional[str] = None

    def feed(self, event: SSEEvent) -> List[StreamChunk]:
        if event.data.strip() == "[DONE]":
            self.done = True
            return self._flush_tools()
        obj = self.load_json(event)
        if obj is None:
            return []
        self.id = self.id or obj.get("id")
        self.resp_model = obj.get("model") or self.resp_model
        chunks: List[StreamChunk] = []
        choices = obj.get("choices") or []
        usage = obj.get("usage") or (choices[0].get("usage") if choices else None)  # Kimi 把 usage 放在 choice 里
        if usage:
            self.usage = parse_openai_usage(usage)
        if not choices:
            return chunks
        choice = choices[0]
        delta = choice.get("delta") or {}
        text = delta.get("content") or ""
        reasoning = _reasoning_of(delta)
        if text or reasoning:
            self.content.append(text)
            self.reasoning.append(reasoning)
            chunks.append(StreamChunk(content=text, reasoning_content=reasoning))
        for i, tc in enumerate(delta.get("tool_calls") or []):
            slot = self.tools.setdefault(tc.get("index", i), {"id": "", "name": "", "arguments": ""})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] += fn["name"]
            if fn.get("arguments"):
                slot["arguments"] += fn["arguments"]
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]
            chunks.extend(self._flush_tools())
            chunks.append(StreamChunk(finish_reason=self.finish_reason))
        return chunks

    @property
    def completed(self) -> bool:
        return self.done or self.finish_reason is not None

    def _tool_calls(self) -> List[ToolCall]:
        return [
            ToolCall(id=v["id"] or f"call_{k}", name=v["name"], arguments=v["arguments"])
            for k, v in sorted(self.tools.items())
        ]

    def _flush_tools(self) -> List[StreamChunk]:
        if getattr(self, "_tools_flushed", False):
            return []
        self._tools_flushed = True
        return [StreamChunk(tool_call=tc) for tc in self._tool_calls()]

    def build(self) -> ChatResponse:
        return ChatResponse(
            content="".join(self.content),
            reasoning_content="".join(self.reasoning),
            tool_calls=self._tool_calls(),
            finish_reason=self.finish_reason,
            usage=self.usage,
            model=self.resp_model or self.model,
            provider=self.provider.name,
            id=self.id,
        )


class AzureOpenAIProvider(OpenAICompatProvider):
    """Azure OpenAI。base_url 形如 https://{resource}.openai.azure.com/openai/deployments/{deployment}
    （model 参数被忽略，由部署名决定）。"""

    api_format = "azure"

    def __init__(self, *args: Any, api_version: str = "2024-10-21", **kwargs: Any):
        kwargs.setdefault("auth_header", "api-key")
        kwargs.setdefault("auth_prefix", "")
        super().__init__(*args, **kwargs)
        self.api_version = api_version

    def build_request(self, messages, model, stream, params) -> RequestSpec:
        spec = super().build_request(messages, model, stream, params)
        spec.query["api-version"] = self.api_version
        return spec
