"""Server-Sent Events 解析（自行按 UTF-8 解码，避免 requests 把无 charset 的流按 ISO-8859-1 解码导致中文乱码）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

import requests


@dataclass
class SSEEvent:
    event: Optional[str]
    data: str


def iter_sse(response: requests.Response, chunk_size: int = 256) -> Iterator[SSEEvent]:
    event: Optional[str] = None
    data_lines: List[str] = []
    buf = b""
    for chunk in response.iter_content(chunk_size=chunk_size):
        if not chunk:
            continue
        buf += chunk
        lines = buf.split(b"\n")
        buf = lines.pop()
        for raw in lines:
            line = raw.rstrip(b"\r").decode("utf-8", errors="replace")
            if line == "":
                if data_lines:
                    yield SSEEvent(event, "\n".join(data_lines))
                event, data_lines = None, []
                continue
            if line.startswith(":"):
                continue  # 注释/心跳
            name, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if name == "data":
                data_lines.append(value)
            elif name == "event":
                event = value
    if buf.strip():
        line = buf.decode("utf-8", errors="replace").strip()
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield SSEEvent(event, "\n".join(data_lines))
