"""流式响应包装。"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterator, List, Optional

import requests

from .providers.base import StreamAccumulator
from .sse import iter_sse
from .types import ChatResponse, StreamChunk, Usage, estimate_messages_tokens, estimate_tokens


class StreamResponse:
    """流式响应。

    用法::

        stream = client.chat("你好", stream=True)
        for chunk in stream:              # StreamChunk
            print(chunk.content, end="")
        print(stream.response.usage)      # 结束后获取完整结果和 token 统计

        # 或只要文本
        for text in client.chat("你好", stream=True).text_stream:
            ...

    只能迭代一次。中途 break 或调用 close() 会断开连接，response.interrupted 为 True。
    """

    def __init__(
        self,
        http_response: requests.Response,
        accumulator: StreamAccumulator,
        request_messages: List[Dict[str, Any]],
        started_at: float,
    ):
        self._http = http_response
        self._acc = accumulator
        self._request_messages = request_messages
        self._started_at = started_at
        self._first_token_at: Optional[float] = None
        self._response: Optional[ChatResponse] = None
        self._started = False
        self._callbacks: List[Callable[[ChatResponse], None]] = []

    # ------------------------------------------------------------------ 迭代
    def __iter__(self) -> Iterator[StreamChunk]:
        if self._started:
            return
        self._started = True
        try:
            for event in iter_sse(self._http):
                for chunk in self._acc.feed(event):
                    if self._first_token_at is None and (chunk.content or chunk.reasoning_content or chunk.tool_call):
                        self._first_token_at = time.time()
                    yield chunk
                if self._acc.done:
                    break
        finally:
            self._finish()

    @property
    def text_stream(self) -> Iterator[str]:
        """只产出正文文本增量。"""
        for chunk in self:
            if chunk.content:
                yield chunk.content

    def until_done(self) -> ChatResponse:
        for _ in self:
            pass
        return self.response

    @property
    def response(self) -> ChatResponse:
        """完整响应（含 usage）。若尚未消费完，会先读完剩余内容。"""
        if self._response is None:
            if not self._started:
                for _ in self:
                    pass
            else:
                self._finish()
        return self._response  # type: ignore[return-value]

    @property
    def usage(self) -> Usage:
        return self.response.usage

    def add_done_callback(self, fn: Callable[[ChatResponse], None]) -> None:
        if self._response is not None:
            fn(self._response)
        else:
            self._callbacks.append(fn)

    # ------------------------------------------------------------------ 结束
    def close(self) -> None:
        self._started = True
        self._finish()

    def _finish(self) -> None:
        if self._response is not None:
            return
        try:
            self._http.close()
        except Exception:  # pragma: no cover
            pass
        resp = self._acc.build()
        now = time.time()
        resp.latency = now - self._started_at
        if self._first_token_at is not None:
            resp.first_token_latency = self._first_token_at - self._started_at
        resp.interrupted = not self._acc.completed
        if resp.usage.is_empty:
            resp.usage = Usage(
                prompt_tokens=estimate_messages_tokens(self._request_messages),
                completion_tokens=estimate_tokens(resp.content) + estimate_tokens(resp.reasoning_content),
                estimated=True,
            )
        self._response = resp
        callbacks, self._callbacks = self._callbacks, []
        for cb in callbacks:
            cb(resp)

    def __enter__(self) -> "StreamResponse":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):  # pragma: no cover
        try:
            if self._response is None:
                self._http.close()
        except Exception:
            pass
