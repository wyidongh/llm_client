"""Google Gemini generateContent 协议。"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

from ..sse import SSEEvent
from ..types import ChatResponse, StreamChunk, ToolCall, Usage
from .base import BaseProvider, RequestSpec, StreamAccumulator

_FINISH_MAP = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter",
               "RECITATION": "content_filter", "PROHIBITED_CONTENT": "content_filter"}


def parse_gemini_usage(u: Optional[Dict[str, Any]]) -> Usage:
    if not u:
        return Usage()
    prompt = u.get("promptTokenCount") or 0
    thoughts = u.get("thoughtsTokenCount") or 0
    completion = (u.get("candidatesTokenCount") or 0) + thoughts
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=u.get("totalTokenCount") or prompt + completion,
        cached_tokens=u.get("cachedContentTokenCount") or 0,
        reasoning_tokens=thoughts,
    )


def _convert_part(part: Any) -> Dict[str, Any]:
    if isinstance(part, str):
        return {"text": part}
    if part.get("type") == "text":
        return {"text": part.get("text", "")}
    if part.get("type") == "image_url":
        url = part["image_url"]["url"] if isinstance(part["image_url"], dict) else part["image_url"]
        if not url.startswith("data:"):
            raise ValueError("Gemini 仅支持 data URL (base64) 形式的图片")
        header, _, data = url.partition(",")
        return {"inline_data": {"mime_type": header[5:].split(";")[0], "data": data}}
    return part  # 已是 Gemini 格式


def _parts(content: Any) -> List[Dict[str, Any]]:
    if content is None or content == "":
        return []
    if isinstance(content, str):
        return [{"text": content}]
    return [_convert_part(p) for p in content]


class GeminiProvider(BaseProvider):
    api_format = "gemini"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"

    def convert_messages(self, messages):
        system: List[str] = []
        contents: List[Dict[str, Any]] = []
        id_to_name: Dict[str, str] = {}

        def push(role, parts):
            if not parts:
                return
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})

        for m in messages:
            role = m.get("role")
            if role in ("system", "developer"):
                c = m.get("content")
                system.append(c if isinstance(c, str) else "".join(p.get("text", "") for p in c))
            elif role == "user":
                push("user", _parts(m.get("content")))
            elif role == "assistant":
                for tc in m.get("tool_calls") or []:
                    id_to_name[tc["id"]] = tc["function"]["name"]
                if m.get("_gemini"):
                    push("model", copy.deepcopy(m["_gemini"]))
                    continue
                parts = _parts(m.get("content"))
                for tc in m.get("tool_calls") or []:
                    fn = tc["function"]
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    parts.append({"functionCall": {"name": fn["name"], "args": args}})
                push("model", parts)
            elif role == "tool":
                name = m.get("name") or id_to_name.get(m.get("tool_call_id"), "tool")
                c = m.get("content")
                try:
                    resp = json.loads(c) if isinstance(c, str) else c
                    if not isinstance(resp, dict):
                        resp = {"result": resp}
                except ValueError:
                    resp = {"result": c}
                push("user", [{"functionResponse": {"name": name, "response": resp}}])
        return "\n\n".join(s for s in system if s), contents

    def build_request(self, messages, model, stream, params) -> RequestSpec:
        params = dict(params)
        system, contents = self.convert_messages(messages)
        body: Dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        gen: Dict[str, Any] = dict(params.pop("generationConfig", None) or {})
        for src, dst in (("temperature", "temperature"), ("top_p", "topP"), ("top_k", "topK"),
                         ("max_tokens", "maxOutputTokens"), ("seed", "seed")):
            if src in params:
                gen[dst] = params.pop(src)
        stop = params.pop("stop", None)
        if stop:
            gen["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)
        rf = params.pop("response_format", None)
        if isinstance(rf, dict) and rf.get("type") == "json_object":
            gen["responseMimeType"] = "application/json"
        params.pop("reasoning_effort", None)  # Gemini 使用 generationConfig.thinkingConfig，请通过 generationConfig 传入
        if gen:
            body["generationConfig"] = gen
        tools = params.pop("tools", None)
        if tools:
            decls = [t["function"] for t in tools if t.get("type") == "function"]
            others = [t for t in tools if t.get("type") != "function"]
            body["tools"] = ([{"functionDeclarations": decls}] if decls else []) + others
        tc = params.pop("tool_choice", None)
        if tc is not None:
            mode = {"auto": "AUTO", "none": "NONE", "required": "ANY"}.get(tc if isinstance(tc, str) else "", None)
            if isinstance(tc, dict) and tc.get("type") == "function":
                body["toolConfig"] = {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [tc["function"]["name"]]}}
            elif mode:
                body["toolConfig"] = {"functionCallingConfig": {"mode": mode}}
        body.update(params)
        headers = {"Content-Type": "application/json"}
        headers.update(self.headers)
        if self.api_key:
            headers["x-goog-api-key"] = self.api_key
        action = "streamGenerateContent" if stream else "generateContent"
        spec = RequestSpec(f"{self.base_url}/models/{model}:{action}", headers, body)
        if stream:
            spec.query["alt"] = "sse"
        return spec

    def response_from_parts(self, parts, finish, usage, model, resp_id, raw=None) -> ChatResponse:
        text, thoughts, tools = [], [], []
        for i, p in enumerate(parts):
            if "functionCall" in p:
                fc = p["functionCall"]
                tools.append(ToolCall(fc.get("id") or f"call_{i}", fc.get("name", ""),
                                      json.dumps(fc.get("args") or {}, ensure_ascii=False)))
            elif "text" in p:
                (thoughts if p.get("thought") else text).append(p["text"])
        finish_reason = "tool_calls" if tools else _FINISH_MAP.get(finish, finish.lower() if finish else None)
        return ChatResponse(
            content="".join(text), reasoning_content="".join(thoughts), tool_calls=tools,
            finish_reason=finish_reason, usage=usage, model=model, provider=self.name, id=resp_id, raw=raw,
            extras={"_gemini": parts} if parts else {},
        )

    def parse_response(self, data, model) -> ChatResponse:
        self.check_error_body(data)
        cand = (data.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        return self.response_from_parts(parts, cand.get("finishReason"), parse_gemini_usage(data.get("usageMetadata")),
                                        data.get("modelVersion") or model, data.get("responseId"), raw=data)

    def stream_accumulator(self, model) -> StreamAccumulator:
        return GeminiStreamAccumulator(self, model)


class GeminiStreamAccumulator(StreamAccumulator):
    def __init__(self, provider: GeminiProvider, model):
        super().__init__(provider, model)
        self.parts: List[Dict[str, Any]] = []
        self.finish: Optional[str] = None
        self.usage_raw: Optional[Dict[str, Any]] = None
        self.id: Optional[str] = None
        self.resp_model: Optional[str] = None

    def _append(self, p: Dict[str, Any]):
        # 合并相邻的同类纯文本 part（不含签名），便于回放
        last = self.parts[-1] if self.parts else None
        if (last is not None and "text" in p and "text" in last and set(p) <= {"text", "thought"}
                and set(last) <= {"text", "thought"} and bool(p.get("thought")) == bool(last.get("thought"))):
            last["text"] += p["text"]
        else:
            self.parts.append(dict(p))

    def feed(self, event: SSEEvent) -> List[StreamChunk]:
        obj = self.load_json(event)
        if obj is None:
            return []
        self.id = self.id or obj.get("responseId")
        self.resp_model = obj.get("modelVersion") or self.resp_model
        if obj.get("usageMetadata"):
            self.usage_raw = obj["usageMetadata"]
        chunks: List[StreamChunk] = []
        cand = (obj.get("candidates") or [{}])[0]
        for i, p in enumerate((cand.get("content") or {}).get("parts") or []):
            self._append(p)
            if "functionCall" in p:
                fc = p["functionCall"]
                chunks.append(StreamChunk(tool_call=ToolCall(fc.get("id") or f"call_{len(self.parts) - 1}",
                                                             fc.get("name", ""),
                                                             json.dumps(fc.get("args") or {}, ensure_ascii=False))))
            elif p.get("text"):
                chunks.append(StreamChunk(reasoning_content=p["text"]) if p.get("thought") else StreamChunk(content=p["text"]))
        if cand.get("finishReason"):
            self.finish = cand["finishReason"]  # 不设置 done：usage 可能在后续 chunk，读到 EOF 为止
            chunks.append(StreamChunk(finish_reason=_FINISH_MAP.get(self.finish, self.finish.lower())))
        return chunks

    @property
    def completed(self) -> bool:
        return self.finish is not None

    def build(self) -> ChatResponse:
        return self.provider.response_from_parts(self.parts, self.finish, parse_gemini_usage(self.usage_raw),
                                                 self.resp_model or self.model, self.id)
