# llm_client

基于 `requests` 的多厂商大模型 API 客户端，唯一依赖是 `requests`，支持 Python 3.8+。

- **多厂商**：内置 OpenAI、Anthropic (Claude)、Google Gemini、Azure OpenAI、DeepSeek、通义千问、Kimi、智谱、豆包、混元、MiniMax、硅基流动、OpenRouter、Groq、Mistral、xAI、Together、Ollama、vLLM、LM Studio
- **自定义厂商**：兼容 OpenAI / Anthropic / Gemini 协议的一行注册；私有协议继承 `BaseProvider`
- **流式和非流式**：用同一个 `chat()`，加 `stream=True` 切换
- **会话**：创建 session、连续对话、截断历史、撤销、分支、保存/加载
- **Token 统计**：单次请求的输入/输出/缓存/推理 token，会话累计，客户端累计；厂商不返回时用本地估算并标记
- **常用功能**：工具调用（OpenAI 格式，自动转换）、思考/推理内容、多模态图片、自动重试（429/5xx，遵循 Retry-After）、超时、代理

## 快速开始

```python
from llm_client import LLMClient

client = LLMClient("deepseek")          # 从 DEEPSEEK_API_KEY 读取 key，也可以传 api_key=
print(client.ask("你好"))
```

### 非流式

```python
r = client.chat("介绍一下你自己", temperature=0.7, max_tokens=512)
print(r.content)            # 正文
print(r.reasoning_content)  # 思考内容（如果有）
print(r.usage)              # 输入=12 输出=85 合计=97
print(r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.cached_tokens)
```

### 流式

```python
stream = client.chat("写一首诗", stream=True)
for chunk in stream:                 # StreamChunk: content / reasoning_content / tool_call / finish_reason
    print(chunk.content, end="", flush=True)
print(stream.response.usage)         # 流结束后拿到完整结果和 token 统计
print(stream.response.first_token_latency)

# 只要文本
for text in client.stream("写一首诗").text_stream:
    print(text, end="")
```

中途 `break` 或调用 `stream.close()` 会断开连接，此时 `stream.response.interrupted == True`。

### 多轮会话

```python
s = client.new_session(system="你是一个助手", temperature=0.5, max_turns=20)
s.chat("我叫小明")
r = s.chat("我叫什么？")                  # 自动携带历史
for t in s.stream("再说一遍").text_stream:   # 流式也会自动写入历史
    print(t, end="")

s.last_usage          # 最近一次请求的 token
s.total_usage         # 本会话累计
s.records             # 每次请求的记录（usage、耗时、finish_reason ……）
s.usage_summary()     # dict 汇总
s.estimate_context_tokens()

s.undo()              # 撤销最后一轮
s2 = s.fork()         # 分支出一个新会话
s.clear()             # 清空历史
s.save("s.json"); s = ChatSession.load(client, "s.json")

client.sessions       # 该客户端创建的所有会话 {id: session}
client.total_usage    # 客户端所有请求的累计 token
```

`max_turns` 只限制发给模型的历史轮数，完整历史一直保留在 `s.messages`。

### 工具调用

```python
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "查询天气",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

s = client.new_session(tools=tools)
r = s.chat("北京天气？")
while r.tool_calls:
    for tc in r.tool_calls:
        s.add_tool_result(tc.id, run_tool(tc.name, tc.parsed_arguments()), name=tc.name)
    r = s.chat()   # 不带参数，让模型继续
```

工具定义统一使用 OpenAI 格式，调用 Anthropic / Gemini 时会自动转换。

## 厂商

```python
from llm_client import list_providers
list_providers()
```

| 名称 | 协议 | Key 环境变量 |
|---|---|---|
| openai | openai | OPENAI_API_KEY |
| anthropic (claude) | anthropic | ANTHROPIC_API_KEY，默认模型 `claude-opus-5-5` |
| gemini (google) | gemini | GEMINI_API_KEY / GOOGLE_API_KEY |
| azure | azure | AZURE_OPENAI_API_KEY，base_url 填到 `.../openai/deployments/{部署名}` |
| deepseek | openai | DEEPSEEK_API_KEY，默认模型 `deepseek-chat` |
| qwen (dashscope) | openai | DASHSCOPE_API_KEY |
| moonshot (kimi) | openai | MOONSHOT_API_KEY |
| zhipu (glm) | openai | ZHIPUAI_API_KEY |
| doubao (ark) | openai | ARK_API_KEY（model 填接入点 ID） |
| hunyuan / minimax / siliconflow / openrouter / groq / mistral / xai / together | openai | `<NAME>_API_KEY` |
| ollama / vllm / lmstudio | openai | 无 |

只有 anthropic 和 deepseek 内置了默认模型，其他厂商需要传 `model=`。

### 自定义厂商

```python
from llm_client import LLMClient, register_provider

# 1) 兼容 OpenAI 协议（最常见）
register_provider("mycorp", base_url="https://llm.mycorp.com/v1", env_key="MYCORP_KEY", default_model="corp-7b")
client = LLMClient("mycorp")

# 2) 不注册，直接用
client = LLMClient("tmp", base_url="https://host/v1", api_key="sk-...", model="xxx")

# 3) 兼容 Anthropic / Gemini 协议的端点
client = LLMClient("proxy", base_url="https://proxy/v1", api_format="anthropic",
                   model="claude-opus-5-5", api_version="2023-06-01")

# 4) 私有协议：继承 BaseProvider，实现 build_request / parse_response / stream_accumulator
register_provider("private", base_url="https://...", provider_class=MyProvider)
```

## 其他参数

```python
LLMClient(
    "openai", model="...", api_key="...", base_url="...",
    timeout=600, connect_timeout=10,     # 流式下 timeout 是两个数据块之间的最长间隔
    max_retries=2,                       # 429/5xx/网络错误自动重试；流式只在收到响应头之前重试
    proxies={"https": "http://127.0.0.1:7890"},
    headers={"X-Custom": "1"},
    default_params={"temperature": 0.3},
)

client.chat(msgs,
    stop=["\n\n"], tools=..., tool_choice="auto",
    reasoning_effort="high",            # OpenAI: reasoning_effort；Anthropic: output_config.effort
    response_format={"type": "json_object"},
    extra_headers={...}, extra_body={...},
    seed=42, presence_penalty=0.1,      # 未列出的参数原样放入请求体
)
```

所有异常都继承自 `LLMError`：`AuthenticationError`、`RateLimitError`、`BadRequestError`、`NotFoundError`、`InternalServerError`、`APITimeoutError`、`APIConnectionError`、`StreamError`。`APIStatusError` 带有 `.status_code`、`.body` 和 `.headers`。

## Anthropic 注意事项

- 默认 `anthropic-version: 2026-10-21`。这个版本下所有模型都不接受 `temperature`、`top_p`、`top_k`，客户端会自动丢弃这些参数。如果端点仍在使用旧版本（比如一些兼容代理），传 `api_version="2023-06-01"` 就会照常发送。
- 在官方端点上使用 `claude-opus-5-5` 等新模型时，默认开启服务端拒答自动回退（`fallbacks: "default"`）。模型拒答（`finish_reason == "refusal"`）后由服务端换模型重试。传 `fallbacks=None` 可以关闭。
- 历史中的 thinking 块（含签名）会原样回放。
- `max_tokens` 默认值：非流式 16000，流式 64000。

## 测试

```bash
python -m unittest discover -s tests -v   # 用本地 mock 服务器模拟三种协议，不需要真实 key
```
