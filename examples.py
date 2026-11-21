"""使用示例。运行前设置对应厂商的 API Key 环境变量，例如:
    export DEEPSEEK_API_KEY=sk-xxx
    python examples.py deepseek
"""
import sys

from llm_client import LLMClient, LLMError, register_provider


def main(provider: str = "deepseek", model: str = None):
    client = LLMClient(provider, model=model)

    # 1. 非流式单轮
    r = client.chat("用一句话介绍你自己")
    print("回复:", r.content)
    print("本次 token:", r.usage, f"耗时 {r.latency:.2f}s")

    # 2. 流式输出
    print("\n流式:")
    stream = client.chat("写一首关于秋天的四行短诗", stream=True)
    for chunk in stream:
        if chunk.reasoning_content:
            print(f"\033[90m{chunk.reasoning_content}\033[0m", end="", flush=True)
        if chunk.content:
            print(chunk.content, end="", flush=True)
    resp = stream.response
    print(f"\n本次 token: {resp.usage}  首字 {resp.first_token_latency:.2f}s")

    # 3. 多轮会话
    session = client.new_session(system="你是一个简洁的助手", temperature=0.7)
    session.chat("我叫小明，喜欢打篮球")
    for text in session.stream("我叫什么？喜欢什么？").text_stream:
        print(text, end="", flush=True)
    print("\n上一轮 token:", session.last_usage)
    print("会话累计 token:", session.total_usage)
    print("客户端累计:", client.total_usage, "请求数", client.request_count)

    # 4. 工具调用
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }]
    s = client.new_session(tools=tools)
    r = s.chat("北京今天天气怎么样？")
    while r.tool_calls:
        for tc in r.tool_calls:
            args = tc.parsed_arguments()
            s.add_tool_result(tc.id, {"city": args.get("city"), "weather": "晴", "temp": 25}, name=tc.name)
        r = s.chat()  # 不带参数：让模型基于工具结果继续
    print("\n工具调用后的回答:", r.content)
    print(s.usage_summary())


def custom_provider_demo():
    # 自定义 OpenAI 兼容厂商
    register_provider("my-llm", base_url="https://llm.example.com/v1", env_key="MY_LLM_KEY", default_model="my-model")
    client = LLMClient("my-llm")
    # 或者一次性使用：LLMClient("anything", base_url="https://...", api_key="...", model="...")
    # Anthropic 协议的代理: LLMClient("proxy", base_url="https://proxy/v1", api_format="anthropic", model="...")
    return client


if __name__ == "__main__":
    try:
        main(*sys.argv[1:3])
    except LLMError as e:
        print("调用失败:", e)
