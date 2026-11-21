"""离线测试：用本地 mock 服务器模拟 OpenAI / Anthropic / Gemini 三种协议。

运行: python -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import (AuthenticationError, ChatSession, LLMClient, RateLimitError,  # noqa: E402
                        register_provider)

STATE = {"requests": [], "fail_next": 0, "reject_stream_options": False}


def sse(handler, events, named=False):
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")  # 故意不带 charset
    handler.send_header("Transfer-Encoding", "chunked")
    handler.end_headers()
    for ev in events:
        if named:
            payload = f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
        else:
            payload = f"data: {ev if isinstance(ev, str) else json.dumps(ev, ensure_ascii=False)}\n\n"
        data = payload.encode("utf-8")
        handler.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        handler.wfile.flush()
    handler.wfile.write(b"0\r\n\r\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        STATE["requests"].append({"path": self.path, "headers": dict(self.headers), "body": body})
        if STATE["fail_next"]:
            STATE["fail_next"] -= 1
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path.startswith("/openai"):
            self.openai(body)
        elif self.path.startswith("/anthropic"):
            self.anthropic(body)
        elif self.path.startswith("/gemini"):
            self.gemini(body)

    # ------------------------------------------------------------- openai
    def openai(self, body):
        if self.headers.get("Authorization") != "Bearer sk-test":
            return self.send_json({"error": {"message": "invalid key"}}, 401)
        if body.get("stream") and "stream_options" in body and STATE["reject_stream_options"]:
            return self.send_json({"error": {"message": "unknown field stream_options"}}, 400)
        n_user = sum(1 for m in body["messages"] if m["role"] == "user")
        last = body["messages"][-1]
        if body.get("tools") and last["role"] != "tool":
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "北京"}'}}]}
            text, finish = "", "tool_calls"
        else:
            text = f"第{n_user}轮回复：你好，世界" if last["role"] != "tool" else "北京晴 25℃"
            msg, finish = {"role": "assistant", "content": text}, "stop"
        usage = {"prompt_tokens": 10 * n_user, "completion_tokens": 7, "total_tokens": 10 * n_user + 7,
                 "prompt_tokens_details": {"cached_tokens": 3}}
        if not body.get("stream"):
            return self.send_json({"id": "x1", "model": body["model"], "choices": [
                {"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})
        events = [{"choices": [{"index": 0, "delta": {"reasoning_content": "思考中"}}]}]
        if finish == "tool_calls":
            events += [
                {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": '{"ci'}}]}}]},
                {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ty": "北京"}'}}]}}]},
            ]
        else:
            events += [{"choices": [{"index": 0, "delta": {"content": ch}}]} for ch in [text[:3], text[3:]]]
        events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        if "stream_options" in body:
            events.append({"choices": [], "usage": usage})
        events.append("[DONE]")
        sse(self, events)

    # ------------------------------------------------------------- anthropic
    def anthropic(self, body):
        assert self.headers.get("x-api-key") == "ak-test"
        assert self.headers.get("anthropic-version")
        n_user = sum(1 for m in body["messages"] if m["role"] == "user")
        content = [{"type": "thinking", "thinking": "", "signature": "sig123"},
                   {"type": "text", "text": f"Claude第{n_user}轮"}]
        usage = {"input_tokens": 20, "output_tokens": 5, "cache_read_input_tokens": 4}
        if not body.get("stream"):
            return self.send_json({"id": "msg_1", "type": "message", "model": body["model"], "content": content,
                                   "stop_reason": "end_turn", "usage": usage})
        events = [
            {"type": "message_start", "message": {"id": "msg_1", "model": body["model"],
                                                  "usage": {"input_tokens": 20, "output_tokens": 1, "cache_read_input_tokens": 4}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig123"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Claude"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": f"第{n_user}轮"}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        ]
        sse(self, events, named=True)

    # ------------------------------------------------------------- gemini
    def gemini(self, body):
        assert self.headers.get("x-goog-api-key") == "gk-test"
        n_user = sum(1 for c in body["contents"] if c["role"] == "user")
        usage = {"promptTokenCount": 12, "candidatesTokenCount": 6, "thoughtsTokenCount": 2, "totalTokenCount": 20}
        if ":generateContent" in self.path:
            return self.send_json({"candidates": [{"content": {"role": "model", "parts": [{"text": f"Gemini第{n_user}轮"}]},
                                                   "finishReason": "STOP"}], "usageMetadata": usage})
        sse(self, [
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "Gemini"}]}}]},
            {"candidates": [{"content": {"role": "model", "parts": [{"text": f"第{n_user}轮"}]}, "finishReason": "STOP"}],
             "usageMetadata": usage},
        ])


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        STATE.update(requests=[], fail_next=0, reject_stream_options=False)

    def openai(self, **kw):
        return LLMClient("openai", model="gpt-test", api_key="sk-test", base_url=self.base + "/openai", **kw)

    # -------------------------------------------------------------- openai
    def test_openai_non_stream(self):
        c = self.openai()
        r = c.chat("hi")
        self.assertEqual(r.content, "第1轮回复：你好，世界")
        self.assertEqual((r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.cached_tokens), (10, 7, 3))
        self.assertEqual(c.total_usage.total_tokens, 17)

    def test_openai_stream(self):
        c = self.openai()
        s = c.chat("hi", stream=True)
        texts = [ch.content for ch in s if ch.content]
        self.assertEqual("".join(texts), "第1轮回复：你好，世界")
        self.assertEqual(s.response.reasoning_content, "思考中")
        self.assertEqual(s.response.usage.total_tokens, 17)
        self.assertFalse(s.response.usage.estimated)
        self.assertFalse(s.response.interrupted)
        self.assertTrue(STATE["requests"][-1]["body"]["stream_options"]["include_usage"])

    def test_stream_options_fallback_and_estimate(self):
        STATE["reject_stream_options"] = True
        c = self.openai()
        r = c.stream("hi").until_done()
        self.assertEqual(r.content, "第1轮回复：你好，世界")
        self.assertTrue(r.usage.estimated)
        self.assertGreater(r.usage.completion_tokens, 0)
        self.assertNotIn("stream_options", STATE["requests"][-1]["body"])

    def test_session_multi_turn(self):
        c = self.openai()
        s = c.new_session(system="你是助手")
        s.chat("你好")
        r = s.chat("再来")
        self.assertEqual(r.content, "第2轮回复：你好，世界")
        sent = STATE["requests"][-1]["body"]["messages"]
        self.assertEqual([m["role"] for m in sent], ["system", "user", "assistant", "user"])
        self.assertEqual(s.total_usage.prompt_tokens, 30)
        self.assertEqual(len(s.records), 2)
        # 流式会话
        st = s.stream("第三轮")
        self.assertEqual("".join(st.text_stream), "第3轮回复：你好，世界")
        self.assertEqual(s.messages[-1]["content"], "第3轮回复：你好，世界")
        self.assertNotIn("_reasoning", STATE["requests"][-1]["body"]["messages"][2])
        # max_turns
        s.max_turns = 1
        self.assertEqual([m["role"] for m in s.context_messages()], ["system", "user", "assistant"])
        # undo / fork / save / load
        f = s.fork()
        s.undo()
        self.assertEqual(s.turns, 2)
        self.assertEqual(f.turns, 3)
        self.assertIn(f.id, c.sessions)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.json")
            s.save(p)
            s2 = ChatSession.load(c, p)
        self.assertEqual(s2.messages, s.messages)
        self.assertEqual(s2.total_usage.total_tokens, s.total_usage.total_tokens)

    def test_tool_calls(self):
        c = self.openai()
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "天气",
                                                   "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
        for stream in (False, True):
            s = c.new_session(tools=tools)
            r = s.chat("北京天气", stream=stream)
            if stream:
                r = r.until_done()
            self.assertEqual(r.finish_reason, "tool_calls")
            self.assertEqual(r.tool_calls[0].parsed_arguments(), {"city": "北京"})
            s.add_tool_result(r.tool_calls[0].id, {"temp": 25})
            r2 = s.chat()
            self.assertEqual(r2.content, "北京晴 25℃")

    def test_retry_and_errors(self):
        STATE["fail_next"] = 2
        c = self.openai(max_retries=2)
        self.assertEqual(c.ask("hi"), "第1轮回复：你好，世界")
        self.assertEqual(len(STATE["requests"]), 3)
        STATE["fail_next"] = 5
        with self.assertRaises(RateLimitError):
            self.openai(max_retries=1).ask("hi")
        STATE["fail_next"] = 0
        bad = LLMClient("openai", model="m", api_key="wrong", base_url=self.base + "/openai")
        with self.assertRaises(AuthenticationError) as cm:
            bad.ask("hi")
        self.assertEqual(cm.exception.message, "invalid key")

    def test_stream_interrupt(self):
        c = self.openai()
        s = c.new_session()
        st = s.stream("hi")
        for ch in st:
            if ch.content:
                break
        st.close()
        self.assertTrue(st.response.interrupted)
        self.assertEqual(s.messages[-1]["role"], "assistant")

    def test_custom_provider(self):
        register_provider("mycorp", base_url=self.base + "/openai", env_key="MYCORP_KEY", default_model="my-model")
        os.environ["MYCORP_KEY"] = "sk-test"
        c = LLMClient("mycorp")
        self.assertEqual(c.ask("hi"), "第1轮回复：你好，世界")
        self.assertEqual(STATE["requests"][-1]["body"]["model"], "my-model")
        # 未注册的厂商直接给 base_url
        c2 = LLMClient("whatever", model="x", api_key="sk-test", base_url=self.base + "/openai")
        self.assertEqual(c2.ask("hi"), "第1轮回复：你好，世界")

    # -------------------------------------------------------------- anthropic
    def test_anthropic(self):
        c = LLMClient("anthropic", api_key="ak-test", base_url=self.base + "/anthropic")
        self.assertEqual(c.model, "claude-opus-5-5")
        s = c.new_session(system="你是助手", temperature=0.5)
        r = s.chat("你好")
        self.assertEqual(r.content, "Claude第1轮")
        self.assertEqual((r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.cached_tokens), (24, 5, 4))
        st = s.stream("继续")
        self.assertEqual("".join(st.text_stream), "Claude第2轮")
        self.assertEqual(st.response.usage.completion_tokens, 5)
        body = STATE["requests"][-1]["body"]
        self.assertEqual(body["system"], "你是助手")
        self.assertNotIn("temperature", body)  # 2026-10-21 版本已移除采样参数
        self.assertNotIn("fallbacks", body)    # 非官方端点不发送
        # thinking 块（含签名）原样回放
        self.assertEqual(body["messages"][1]["content"][0], {"type": "thinking", "thinking": "", "signature": "sig123"})
        s.chat("第三")
        self.assertEqual(STATE["requests"][-1]["body"]["messages"][3]["content"][0]["signature"], "sig123")

    def test_anthropic_request_mapping(self):
        from llm_client import AnthropicProvider
        p = AnthropicProvider(api_key="k")
        spec = p.build_request(
            [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": None, "tool_calls": [
                 {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{\"a\":1}"}},
                 {"id": "t2", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "t1", "content": "r1"},
             {"role": "tool", "tool_call_id": "t2", "content": "r2"}],
            "claude-opus-5-5", False, {"reasoning_effort": "high", "stop": "END"})
        b = spec.body
        self.assertEqual(b["fallbacks"], "default")
        self.assertEqual(spec.headers["anthropic-beta"], "server-side-fallback-2026-07-01")
        self.assertEqual(b["output_config"], {"effort": "high"})
        self.assertEqual(b["stop_sequences"], ["END"])
        self.assertEqual(len(b["messages"]), 3)  # 两个 tool 结果合并到同一条 user 消息
        self.assertEqual([x["tool_use_id"] for x in b["messages"][2]["content"]], ["t1", "t2"])

    # -------------------------------------------------------------- gemini
    def test_gemini(self):
        c = LLMClient("gemini", model="gemini-test", api_key="gk-test", base_url=self.base + "/gemini")
        s = c.new_session(system="sys", max_tokens=100)
        r = s.chat("你好")
        self.assertEqual(r.content, "Gemini第1轮")
        self.assertEqual((r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.reasoning_tokens), (12, 8, 2))
        st = s.stream("继续")
        self.assertEqual("".join(st.text_stream), "Gemini第2轮")
        self.assertFalse(st.response.interrupted)
        req = STATE["requests"][-1]
        self.assertIn(":streamGenerateContent?alt=sse", req["path"])
        self.assertEqual(req["body"]["generationConfig"]["maxOutputTokens"], 100)
        self.assertEqual([c["role"] for c in req["body"]["contents"]], ["user", "model", "user"])
        self.assertEqual(s.total_usage.total_tokens, 40)


if __name__ == "__main__":
    unittest.main()
