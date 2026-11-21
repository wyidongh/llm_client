"""统一异常体系。"""
from __future__ import annotations

from typing import Any, Mapping, Optional


class LLMError(Exception):
    """所有异常的基类。"""


class APIConnectionError(LLMError):
    """网络连接失败（DNS、拒绝连接、SSL 等）。"""


class APITimeoutError(APIConnectionError):
    """请求超时。"""


class StreamError(LLMError):
    """流式响应过程中服务端返回了错误事件。"""


class APIStatusError(LLMError):
    """服务端返回了 HTTP 错误状态码（或 200 但 body 中带 error）。"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        body: Any = None,
        headers: Optional[Mapping[str, str]] = None,
    ):
        super().__init__(f"[{status_code}] {message}" if status_code else message)
        self.message = message
        self.status_code = status_code
        self.body = body
        self.headers = dict(headers or {})


class BadRequestError(APIStatusError):
    """400"""


class AuthenticationError(APIStatusError):
    """401"""


class PermissionDeniedError(APIStatusError):
    """403"""


class NotFoundError(APIStatusError):
    """404"""


class RateLimitError(APIStatusError):
    """429"""


class InternalServerError(APIStatusError):
    """>= 500"""


_STATUS_MAP = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    422: BadRequestError,
    429: RateLimitError,
}


def make_status_error(status_code: int, message: str, body: Any = None, headers=None) -> APIStatusError:
    cls = _STATUS_MAP.get(status_code)
    if cls is None:
        cls = InternalServerError if status_code >= 500 else APIStatusError
    return cls(message, status_code=status_code, body=body, headers=headers)


def is_retryable_status(status_code: int) -> bool:
    return status_code in (408, 409, 425, 429) or status_code >= 500
