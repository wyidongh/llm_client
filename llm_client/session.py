"""ChatSession：多轮连续对话 + 每轮 / 累计 token 统计。"""
from __future__ import annotations

import copy
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .stream import StreamResponse
from .types import ChatResponse, Usage, estimate_messages_tokens

if TYPE_CHECKING:
    from .client import LLMClient


@dataclass
class TurnRecord:
    """一次请求的记录。"""

    index: int
    model: Optional[str]
    usage: Usage
    finish_reason: Optional[str]
    latency: Optional[float]
    first_token_latency: Optional[float] = None
    stream: bool = False
    interrupted: bool = False
    timestamp: float = field(default_factory=time.time)


class ChatSession:
    """多轮对话会话。会自动维护历史消息，并统计每次请求与整个会话的 token 用量。

    Args:
        client: LLMClient
        system: 系统提示词
        session_id: 会话 ID，不传则自动生成
        max_turns: 发送给模型的最大历史轮数（一轮 = 一条用户消息及其后的回复/工具消息），None 表示不限制。
                   完整历史仍保留在 self.messages 中。
        **default_params: 本会话每次请求的默认参数（model、temperature、tools 等）
    """

    def __init__(
        self,
        client: "LLMClient",
        system: Optional[str] = None,
        session_id: Optional[str] = None,
        max_turns: Optional[int] = None,
        **default_params: Any,
    ):
        self.client = client
        self.id = session_id or uuid.uuid4().hex
        self.system = system
        self.max_turns = max_turns
        self.default_params = default_params
        self.messages: List[Dict[str, Any]] = []
        self.records: List[TurnRecord] = []
        self.total_usage = Usage()
        self.created_at = time.time()

    # ================================================================== 对话
    def chat(self, content: Any = None, stream: bool = False, **params: Any) -> Union[ChatResponse, StreamResponse]:
        """发送一条用户消息并获取回复，回复会自动加入历史。

        content 可以是字符串或多模态 content 列表；为 None 时不追加用户消息
        （用于在 add_tool_result() 之后让模型继续）。
        """
        added = False
        if content is not None:
            self.messages.append({"role": "user", "content": content})
            added = True
        req_params = {**self.default_params, **params}
        try:
            result = self.client.chat(self.context_messages(), stream=stream, **req_params)
        except Exception:
            if added:
                self.messages.pop()
            raise
        if isinstance(result, StreamResponse):
            result.add_done_callback(lambda r: self._on_response(r, stream=True, user_added=added))
        else:
            self._on_response(result, stream=False, user_added=added)
        return result

    def stream(self, content: Any = None, **params: Any) -> StreamResponse:
        """流式发送，等价于 chat(content, stream=True)。流结束（或中断）后回复自动加入历史。"""
        return self.chat(content, stream=True, **params)  # type: ignore[return-value]

    def ask(self, content: Any, **params: Any) -> str:
        return self.chat(content, stream=False, **params).content  # type: ignore[union-attr]

    def _on_response(self, resp: ChatResponse, stream: bool, user_added: bool) -> None:
        if resp.content or resp.tool_calls or resp.extras:
            self.messages.append(resp.to_message())
        elif user_added and self.messages and self.messages[-1].get("role") == "user":
            # 没有任何输出（如连接立即中断），撤回用户消息，避免历史中出现连续两条 user 消息
            self.messages.pop()
        self.records.append(TurnRecord(
            index=len(self.records) + 1, model=resp.model, usage=resp.usage, finish_reason=resp.finish_reason,
            latency=resp.latency, first_token_latency=resp.first_token_latency, stream=stream,
            interrupted=resp.interrupted,
        ))
        self.total_usage = self.total_usage + resp.usage

    # ================================================================== 历史管理
    def context_messages(self) -> List[Dict[str, Any]]:
        """实际发送给模型的消息（system + 截断后的历史）。"""
        history = self.messages
        if self.max_turns:
            starts = [i for i, m in enumerate(history) if m.get("role") == "user"]
            if len(starts) > self.max_turns:
                history = history[starts[-self.max_turns]:]
        msgs = [{"role": "system", "content": self.system}] if self.system else []
        return msgs + list(history)

    def add_message(self, role: str, content: Any, **extra: Any) -> None:
        """手动追加消息（如注入示例对话）。"""
        self.messages.append({"role": role, "content": content, **extra})

    def add_tool_result(self, tool_call_id: str, content: Any, name: Optional[str] = None,
                        is_error: bool = False) -> None:
        """追加工具执行结果。之后调用 chat() 不带参数让模型继续。"""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        msg: Dict[str, Any] = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        if name:
            msg["name"] = name
        if is_error:
            msg["is_error"] = True
        self.messages.append(msg)

    def undo(self) -> List[Dict[str, Any]]:
        """撤销最后一轮（最后一条用户消息及其之后的所有消息），返回被移除的消息。"""
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "user":
                removed = self.messages[i:]
                self.messages = self.messages[:i]
                return removed
        return []

    def clear(self, reset_usage: bool = False) -> None:
        """清空历史（保留 system）。"""
        self.messages = []
        if reset_usage:
            self.records = []
            self.total_usage = Usage()

    def fork(self, session_id: Optional[str] = None) -> "ChatSession":
        """复制当前会话为一个新会话（分支对话）。"""
        s = self.client.new_session(system=self.system, session_id=session_id, max_turns=self.max_turns,
                                    **self.default_params)
        s.messages = copy.deepcopy(self.messages)
        return s

    # ================================================================== 统计
    @property
    def last_usage(self) -> Optional[Usage]:
        return self.records[-1].usage if self.records else None

    @property
    def turns(self) -> int:
        return sum(1 for m in self.messages if m.get("role") == "user")

    def estimate_context_tokens(self) -> int:
        """估算下一次请求的上下文 token 数（不含新消息）。"""
        return estimate_messages_tokens(self.context_messages())

    def usage_summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.id,
            "requests": len(self.records),
            "turns": self.turns,
            "total": self.total_usage.to_dict(),
            "last": self.last_usage.to_dict() if self.last_usage else None,
            "per_request": [r.usage.to_dict() for r in self.records],
        }

    # ================================================================== 持久化
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "system": self.system,
            "max_turns": self.max_turns,
            "default_params": self.default_params,
            "messages": self.messages,
            "records": [asdict(r) for r in self.records],
            "created_at": self.created_at,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, client: "LLMClient", data: Dict[str, Any]) -> "ChatSession":
        s = cls(client, system=data.get("system"), session_id=data.get("id"), max_turns=data.get("max_turns"),
                **(data.get("default_params") or {}))
        s.messages = data.get("messages") or []
        for r in data.get("records") or []:
            r = dict(r)
            r["usage"] = Usage(**r["usage"])
            rec = TurnRecord(**r)
            s.records.append(rec)
            s.total_usage = s.total_usage + rec.usage
        s.created_at = data.get("created_at", s.created_at)
        client.sessions[s.id] = s
        return s

    @classmethod
    def load(cls, client: "LLMClient", path: str) -> "ChatSession":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(client, json.load(f))

    def __repr__(self) -> str:
        return f"<ChatSession id={self.id[:8]} turns={self.turns} tokens={self.total_usage.total_tokens}>"
