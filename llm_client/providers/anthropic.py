"""Anthropic Messages API 协议（Claude，以及兼容 Anthropic 协议的第三方端点）。"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

from ..sse import SSEEvent
from ..types import ChatResponse, StreamChunk, ToolCall, Usage
from .base import BaseProvider, RequestSpec, StreamAccumulator

OFFICIAL_BASE_URL = "https://api.anthropic.com/v1"
# 这些模型可能以 stop_reason="refusal" 拒答；官方端点上默认开启服务端自动回退
FALLBACK_MODELS = ("claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# 自 2026-10-21 版本起，temperature/top_p/top_k 在所有模型上都会返回 400
SAMPLING_REMOVED_VERSION = "2026-10-21"

_STOP_MAP = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls"}


def parse_anthropic_usage(u: Optional[Dict[str, Any]]) -> Usage:
    if not u:
        return Usage()
    cache_read = u.get("cache_read_input_tokens") or 0
    cache_write = u.get("cache_creation_input_tokens") or 0
    prompt = (u.get("input_tokens") or 0) + cache_read + cache_write
    completion = u.get("output_tokens") or 0
    return Usage(prompt_tokens=prompt, completion_tokens=completion, cached_tokens=cache_read)


def _convert_part(part: Any) -> Any:
    """OpenAI 风格的多模态 part -> Anthropic content block；其他格式原样透传。"""
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        return part
    if part.get("type") == "image_url":
        url = part["image_url"]["url"] if isinstance(part["image_url"], dict) else part["image_url"]
        if url.startswith("data:"):
            header, _, data = url.partition(",")
            media_type = header[5:].split(";")[0]
            return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
        return {"type": "image", "source": {"type": "url", "url": url}}
    return part


def _as_blocks(content: Any) -> List[Dict[str, Any]]:
    if content is None or content == "":
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [_convert_part(p) for p in content]


class AnthropicProvider(BaseProvider):
    api_format = "anthropic"
    default_base_url = OFFICIAL_BASE_URL

    def __init__(
        self,
        *args: Any,
        api_version: str = SAMPLING_REMOVED_VERSION,
        fallbacks: Any = "auto",
        auth_header: str = "x-api-key",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.api_version = api_version
        # "auto": 官方端点 + 支持的模型时发送 fallbacks="default"；None/False 关闭；其他值原样发送
        self.fallbacks = fallbacks
        self.auth_header = auth_header

    # ---------------------------------------------------------------- 消息转换
    def convert_messages(self, messages: List[Dict[str, Any]]):
        system_parts: List[str] = []
        out: List[Dict[str, Any]] = []

        def push(role: str, blocks: List[Dict[str, Any]]):
            if not blocks:
                return
            if out and out[-1]["role"] == role:  # 合并相邻同角色消息（如多个工具结果）
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": blocks})

        for m in messages:
            role = m.get("role")
            if role in ("system", "developer"):
                c = m.get("content")
                system_parts.append(c if isinstance(c, str) else "".join(p.get("text", "") for p in c))
            elif role == "user":
                push("user", _as_blocks(m.get("content")))
            elif role == "assistant":
                if m.get("_anthropic"):  # 原样回放（保留 thinking 签名等）
                    push("assistant", copy.deepcopy(m["_anthropic"]))
                    continue
                blocks = _as_blocks(m.get("content"))
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": fn.get("name"), "input": args})
                push("assistant", blocks)
            elif role == "tool":
                c = m.get("content")
                block = {
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id"),
                    "content": c if isinstance(c, str) else _as_blocks(c),
                }
                if m.get("is_error"):
                    block["is_error"] = True
                push("user", [block])
        return "\n\n".join(p for p in system_parts if p), out

    @staticmethod
    def convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for t in tools:
            if t.get("type") == "function" and "function" in t:
                fn = t["function"]
                out.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
                })
            else:
                out.append(t)  # 已是 Anthropic 格式（含服务端工具）
        return out

    @staticmethod
    def convert_tool_choice(tc: Any) -> Any:
        if tc == "auto":
            return {"type": "auto"}
        if tc == "none":
            return {"type": "none"}
        if tc == "required":
            return {"type": "any"}
        if isinstance(tc, dict) and tc.get("type") == "function":
            return {"type": "tool", "name": tc["function"]["name"]}
        return tc

    # ---------------------------------------------------------------- 请求
    def _use_fallbacks(self, model: str) -> Any:
        if self.fallbacks == "auto":
            if self.base_url == OFFICIAL_BASE_URL and model in FALLBACK_MODELS:
                return "default"
            return None
        return self.fallbacks or None

    def build_request(self, messages, model, stream, params) -> RequestSpec:
        params = dict(params)
        system, msgs = self.convert_messages(messages)
        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": params.pop("max_tokens", None) or (64000 if stream else 16000),
            "messages": msgs,
        }
        if system:
            body["system"] = system
        sampling = {k: params.pop(k) for k in ("temperature", "top_p", "top_k") if k in params}
        if self.api_version < SAMPLING_REMOVED_VERSION:
            body.update(sampling)
        stop = params.pop("stop", None)
        if stop:
            body["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
        tools = params.pop("tools", None)
        if tools:
            body["tools"] = self.convert_tools(tools)
        tool_choice = params.pop("tool_choice", None)
        if tool_choice is not None:
            body["tool_choice"] = self.convert_tool_choice(tool_choice)
        effort = params.pop("reasoning_effort", None)
        if effort is not None:
            body.setdefault("output_config", {})["effort"] = effort
        body.update(params)  # 其他参数（thinking、metadata、cache_control 等）原样透传
        if stream:
            body["stream"] = True

        headers = {"Content-Type": "application/json", "anthropic-version": self.api_version}
        fb = self._use_fallbacks(model)
        if fb is not None and "fallbacks" not in body:
            body["fallbacks"] = fb
            headers["anthropic-beta"] = FALLBACK_BETA
        for k, v in self.headers.items():
            if k.lower() == "anthropic-beta" and "anthropic-beta" in headers:
                headers["anthropic-beta"] = headers["anthropic-beta"] + "," + v
            else:
                headers[k] = v
        if self.api_key:
            headers[self.auth_header] = self.api_key if self.auth_header == "x-api-key" else f"Bearer {self.api_key}"
        return RequestSpec(self.base_url + "/messages", headers, body)

    # ---------------------------------------------------------------- 响应
    def response_from_blocks(self, blocks, stop_reason, usage, model, msg_id, raw=None) -> ChatResponse:
        texts, thinking, tools = [], [], []
        for b in blocks:
            t = b.get("type")
            if t == "text":
                texts.append(b.get("text", ""))
            elif t == "thinking":
                thinking.append(b.get("thinking", ""))
            elif t == "tool_use":
                tools.append(ToolCall(b["id"], b["name"], json.dumps(b.get("input") or {}, ensure_ascii=False)))
        return ChatResponse(
            content="".join(texts),
            reasoning_content="".join(thinking),
            tool_calls=tools,
            finish_reason=_STOP_MAP.get(stop_reason, stop_reason),
            usage=usage,
            model=model,
            provider=self.name,
            id=msg_id,
            raw=raw,
            extras={"_anthropic": blocks} if blocks else {},
        )

    def parse_response(self, data, model) -> ChatResponse:
        if data.get("type") == "error":
            self.check_error_body(data)
        return self.response_from_blocks(
            data.get("content") or [], data.get("stop_reason"), parse_anthropic_usage(data.get("usage")),
            data.get("model") or model, data.get("id"), raw=data,
        )

    def stream_accumulator(self, model) -> StreamAccumulator:
        return AnthropicStreamAccumulator(self, model)


class AnthropicStreamAccumulator(StreamAccumulator):
    def __init__(self, provider: AnthropicProvider, model):
        super().__init__(provider, model)
        self.blocks: Dict[int, Dict[str, Any]] = {}
        self.json_buf: Dict[int, str] = {}
        self.usage_raw: Dict[str, Any] = {}
        self.stop_reason: Optional[str] = None
        self.id: Optional[str] = None
        self.resp_model: Optional[str] = None

    def feed(self, event: SSEEvent) -> List[StreamChunk]:
        obj = self.load_json(event)
        if obj is None:
            return []
        t = obj.get("type")
        if t == "message_start":
            msg = obj.get("message") or {}
            self.id, self.resp_model = msg.get("id"), msg.get("model")
            self.usage_raw.update(msg.get("usage") or {})
        elif t == "content_block_start":
            idx = obj["index"]
            block = dict(obj.get("content_block") or {})
            if block.get("type") == "tool_use":
                block["input"] = {}
                self.json_buf[idx] = ""
            self.blocks[idx] = block
        elif t == "content_block_delta":
            idx, d = obj["index"], obj.get("delta") or {}
            block = self.blocks.setdefault(idx, {"type": "text", "text": ""})
            dt = d.get("type")
            if dt == "text_delta":
                block["text"] = block.get("text", "") + d.get("text", "")
                return [StreamChunk(content=d.get("text", ""))]
            if dt == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + d.get("thinking", "")
                return [StreamChunk(reasoning_content=d.get("thinking", ""))]
            if dt == "signature_delta":
                block["signature"] = block.get("signature", "") + d.get("signature", "")
            elif dt == "input_json_delta":
                self.json_buf[idx] = self.json_buf.get(idx, "") + d.get("partial_json", "")
            elif dt == "citations_delta":
                block.setdefault("citations", []).append(d.get("citation"))
        elif t == "content_block_stop":
            idx = obj["index"]
            block = self.blocks.get(idx) or {}
            if block.get("type") == "tool_use":
                try:
                    block["input"] = json.loads(self.json_buf.get(idx) or "{}")
                except ValueError:
                    block["input"] = {}
                return [StreamChunk(tool_call=ToolCall(block["id"], block["name"],
                                                       json.dumps(block["input"], ensure_ascii=False)))]
        elif t == "message_delta":
            self.stop_reason = (obj.get("delta") or {}).get("stop_reason") or self.stop_reason
            self.usage_raw.update({k: v for k, v in (obj.get("usage") or {}).items() if v is not None})
            if self.stop_reason:
                return [StreamChunk(finish_reason=_STOP_MAP.get(self.stop_reason, self.stop_reason))]
        elif t == "message_stop":
            self.done = True
        elif t == "error":
            from ..errors import StreamError
            raise StreamError(f"stream error: {(obj.get('error') or {}).get('message')}")
        return []

    def build(self) -> ChatResponse:
        blocks = [self.blocks[i] for i in sorted(self.blocks)]
        return self.provider.response_from_blocks(
            blocks, self.stop_reason, parse_anthropic_usage(self.usage_raw),
            self.resp_model or self.model, self.id,
        )
