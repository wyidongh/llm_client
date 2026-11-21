"""LLMClient：统一入口。"""
from __future__ import annotations

import copy
import json
import logging
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from .errors import (APIConnectionError, APIStatusError, APITimeoutError, BadRequestError, is_retryable_status,
                     make_status_error)
from .providers.base import BaseProvider, RequestSpec
from .providers.registry import create_provider
from .stream import StreamResponse
from .types import ChatResponse, Usage, estimate_messages_tokens, estimate_tokens

logger = logging.getLogger("llm_client")

Messages = Union[str, Dict[str, Any], List[Dict[str, Any]]]


def normalize_messages(messages: Messages) -> List[Dict[str, Any]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if isinstance(messages, dict):
        return [messages]
    return list(messages)


class LLMClient:
    """统一的大模型客户端。

    Args:
        provider: 厂商名（见 list_providers()）、自定义注册名、或 BaseProvider 实例
        model: 默认模型
        api_key: 不传时从厂商对应的环境变量读取
        base_url: 覆盖默认地址（用于代理/私有部署/自定义厂商）
        api_format: 未注册的自定义厂商使用的协议：openai / anthropic / gemini
        timeout: 读超时秒数（流式为两个数据块之间的最大间隔）
        connect_timeout: 连接超时秒数
        max_retries: 对 408/409/429/5xx 及网络错误的最大重试次数（流式仅在收到响应头前重试）
        proxies / verify: 透传给 requests
        headers: 额外请求头
        default_params: 每次请求的默认参数，例如 temperature=0.7
        **provider_options: 传给 Provider 构造器的额外选项（如 Anthropic 的 api_version、Azure 的 api_version）
    """

    def __init__(
        self,
        provider: Any = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        api_format: Optional[str] = None,
        timeout: float = 600,
        connect_timeout: float = 10,
        max_retries: int = 2,
        proxies: Optional[Dict[str, str]] = None,
        verify: Union[bool, str] = True,
        headers: Optional[Dict[str, str]] = None,
        default_params: Optional[Dict[str, Any]] = None,
        http_session: Optional[requests.Session] = None,
        **provider_options: Any,
    ):
        self.provider: BaseProvider = create_provider(
            provider, api_key=api_key, base_url=base_url, default_model=model, headers=headers,
            api_format=api_format, **provider_options,
        )
        self.model = model or self.provider.default_model
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.max_retries = max_retries
        self.proxies = proxies
        self.verify = verify
        self.default_params = dict(default_params or {})
        self.http = http_session or requests.Session()
        self.sessions: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self.total_usage = Usage()
        self.request_count = 0
        self.last_usage: Optional[Usage] = None

    # ================================================================== 对话
    def chat(
        self,
        messages: Messages,
        stream: bool = False,
        model: Optional[str] = None,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Union[str, List[str], None] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        reasoning_effort: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: Union[float, Tuple[float, float], None] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[ChatResponse, StreamResponse]:
        """发送一次对话请求（无状态）。

        messages 可以是字符串、单条消息 dict 或 OpenAI 风格的消息列表。
        stream=False 返回 ChatResponse；stream=True 返回可迭代的 StreamResponse。
        tools 使用 OpenAI 格式，会自动转换为 Anthropic / Gemini 格式。
        未列出的关键字参数原样放入请求体（如 seed、presence_penalty、thinking 等）。
        """
        msgs = normalize_messages(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        model = model or self.model
        if not model:
            raise ValueError(f"provider '{self.provider.name}' 没有默认模型，请传入 model")

        params: Dict[str, Any] = dict(self.default_params)
        params.update(dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop, tools=tools,
                           tool_choice=tool_choice, reasoning_effort=reasoning_effort,
                           response_format=response_format))
        params.update(kwargs)
        params = {k: v for k, v in params.items() if v is not None}

        def build() -> RequestSpec:
            spec = self.provider.build_request(msgs, model, stream, params)
            if extra_headers:
                spec.headers.update(extra_headers)
            if extra_body:
                spec.body.update(copy.deepcopy(extra_body))
            return spec

        started = time.time()
        try:
            http_resp = self._send(build(), stream, timeout)
        except BadRequestError as e:
            # 部分 OpenAI 兼容厂商不支持 stream_options，自动关闭后重试一次
            if stream and getattr(self.provider, "stream_usage", False) and "stream_options" in str(e).lower():
                logger.info("provider %s 不支持 stream_options，已自动关闭", self.provider.name)
                self.provider.stream_usage = False  # type: ignore[attr-defined]
                http_resp = self._send(build(), stream, timeout)
            else:
                raise

        if stream:
            sr = StreamResponse(http_resp, self.provider.stream_accumulator(model), msgs, started)
            sr.add_done_callback(self._record)
            return sr

        try:
            data = json.loads(http_resp.content.decode("utf-8"))
        except ValueError as e:
            raise APIStatusError(f"无法解析响应 JSON: {http_resp.text[:500]}", http_resp.status_code) from e
        resp = self.provider.parse_response(data, model)
        resp.latency = time.time() - started
        if resp.usage.is_empty:
            resp.usage = Usage(
                prompt_tokens=estimate_messages_tokens(msgs),
                completion_tokens=estimate_tokens(resp.content) + estimate_tokens(resp.reasoning_content),
                estimated=True,
            )
        self._record(resp)
        return resp

    def stream(self, messages: Messages, **kwargs: Any) -> StreamResponse:
        """等价于 chat(messages, stream=True)。"""
        return self.chat(messages, stream=True, **kwargs)  # type: ignore[return-value]

    def ask(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> str:
        """单轮问答，直接返回文本。"""
        return self.chat(prompt, system=system, stream=False, **kwargs).content  # type: ignore[union-attr]

    # ================================================================== 会话
    def new_session(self, system: Optional[str] = None, session_id: Optional[str] = None, **kwargs: Any):
        """创建一个新的多轮对话会话（ChatSession）。kwargs 为该会话的默认请求参数或 max_turns。"""
        from .session import ChatSession

        s = ChatSession(self, system=system, session_id=session_id, **kwargs)
        self.sessions[s.id] = s
        return s

    def get_session(self, session_id: str):
        return self.sessions[session_id]

    def remove_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    # ================================================================== 统计
    def _record(self, resp: ChatResponse) -> None:
        with self._lock:
            self.total_usage = self.total_usage + resp.usage
            self.request_count += 1
            self.last_usage = resp.usage

    def reset_usage(self) -> None:
        with self._lock:
            self.total_usage = Usage()
            self.request_count = 0
            self.last_usage = None

    # ================================================================== HTTP
    def _send(self, spec: RequestSpec, stream: bool, timeout) -> requests.Response:
        if timeout is None:
            timeout = (self.connect_timeout, self.timeout)
        payload = json.dumps(spec.body, ensure_ascii=False).encode("utf-8")
        attempt = 0
        while True:
            err: Exception
            retry_after: Optional[float] = None
            try:
                resp = self.http.post(spec.url, params=spec.query or None, headers=spec.headers, data=payload,
                                      stream=stream, timeout=timeout, proxies=self.proxies, verify=self.verify)
            except requests.Timeout as e:
                err = APITimeoutError(f"请求超时: {e}")
            except requests.RequestException as e:
                err = APIConnectionError(f"连接失败: {e}")
            else:
                if resp.status_code < 400:
                    return resp
                err = self._status_error(resp)
                resp.close()
                if not is_retryable_status(resp.status_code) or resp.headers.get("x-should-retry") == "false":
                    raise err
                retry_after = _parse_retry_after(resp.headers.get("retry-after"))
            if attempt >= self.max_retries:
                raise err
            delay = retry_after if retry_after is not None else min(0.5 * 2 ** attempt, 8) * (0.75 + random.random() / 2)
            logger.warning("请求失败（%s），%.1fs 后第 %d 次重试", err, delay, attempt + 1)
            time.sleep(delay)
            attempt += 1

    @staticmethod
    def _status_error(resp: requests.Response) -> APIStatusError:
        text = resp.content.decode("utf-8", errors="replace")
        body: Any = text
        message = text[:1000] or resp.reason
        try:
            body = json.loads(text)
            err = body.get("error", body) if isinstance(body, dict) else body
            if isinstance(err, dict):
                message = err.get("message") or err.get("msg") or json.dumps(err, ensure_ascii=False)
            elif isinstance(err, str):
                message = err
            if isinstance(body, list) and body and isinstance(body[0], dict):  # Gemini 有时返回数组
                message = (body[0].get("error") or {}).get("message", message)
        except ValueError:
            pass
        return make_status_error(resp.status_code, message, body, resp.headers)

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<LLMClient provider={self.provider.name!r} model={self.model!r}>"


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return min(float(value), 60.0)
    except ValueError:
        return None
