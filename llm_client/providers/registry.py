"""厂商预设与注册表。"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Type

from .anthropic import AnthropicProvider
from .base import BaseProvider
from .gemini import GeminiProvider
from .openai_compat import AzureOpenAIProvider, OpenAICompatProvider

# api_format -> Provider 类
FORMATS: Dict[str, Type[BaseProvider]] = {
    "openai": OpenAICompatProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "azure": AzureOpenAIProvider,
}

# 内置厂商预设。default_model 为 None 的需要在调用时传 model。
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai":      dict(api_format="openai", base_url="https://api.openai.com/v1", env_key="OPENAI_API_KEY"),
    "anthropic":   dict(api_format="anthropic", base_url="https://api.anthropic.com/v1", env_key="ANTHROPIC_API_KEY",
                        default_model="claude-opus-5-5"),
    "gemini":      dict(api_format="gemini", base_url="https://generativelanguage.googleapis.com/v1beta",
                        env_key=("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    "azure":       dict(api_format="azure", base_url=None, env_key="AZURE_OPENAI_API_KEY"),
    "deepseek":    dict(api_format="openai", base_url="https://api.deepseek.com/v1", env_key="DEEPSEEK_API_KEY",
                        default_model="deepseek-chat"),
    "qwen":        dict(api_format="openai", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        env_key="DASHSCOPE_API_KEY"),
    "moonshot":    dict(api_format="openai", base_url="https://api.moonshot.cn/v1", env_key="MOONSHOT_API_KEY"),
    "zhipu":       dict(api_format="openai", base_url="https://open.bigmodel.cn/api/paas/v4", env_key="ZHIPUAI_API_KEY"),
    "doubao":      dict(api_format="openai", base_url="https://ark.cn-beijing.volces.com/api/v3", env_key="ARK_API_KEY"),
    "hunyuan":     dict(api_format="openai", base_url="https://api.hunyuan.cloud.tencent.com/v1", env_key="HUNYUAN_API_KEY"),
    "minimax":     dict(api_format="openai", base_url="https://api.minimaxi.com/v1", env_key="MINIMAX_API_KEY"),
    "siliconflow": dict(api_format="openai", base_url="https://api.siliconflow.cn/v1", env_key="SILICONFLOW_API_KEY"),
    "openrouter":  dict(api_format="openai", base_url="https://openrouter.ai/api/v1", env_key="OPENROUTER_API_KEY"),
    "groq":        dict(api_format="openai", base_url="https://api.groq.com/openai/v1", env_key="GROQ_API_KEY"),
    "mistral":     dict(api_format="openai", base_url="https://api.mistral.ai/v1", env_key="MISTRAL_API_KEY"),
    "xai":         dict(api_format="openai", base_url="https://api.x.ai/v1", env_key="XAI_API_KEY"),
    "together":    dict(api_format="openai", base_url="https://api.together.xyz/v1", env_key="TOGETHER_API_KEY"),
    "ollama":      dict(api_format="openai", base_url="http://localhost:11434/v1", env_key=None),
    "vllm":        dict(api_format="openai", base_url="http://localhost:8000/v1", env_key=None),
    "lmstudio":    dict(api_format="openai", base_url="http://localhost:1234/v1", env_key=None),
}

ALIASES = {
    "claude": "anthropic", "google": "gemini", "azure_openai": "azure",
    "dashscope": "qwen", "tongyi": "qwen", "aliyun": "qwen",
    "kimi": "moonshot", "glm": "zhipu", "zhipuai": "zhipu", "bigmodel": "zhipu",
    "ark": "doubao", "volcengine": "doubao", "silicon": "siliconflow", "grok": "xai",
}


def register_format(name: str, provider_class: Type[BaseProvider]) -> None:
    """注册一种新的协议格式（自定义 Provider 类）。"""
    FORMATS[name.lower()] = provider_class


def register_provider(
    name: str,
    base_url: Optional[str] = None,
    api_format: str = "openai",
    env_key: Any = None,
    default_model: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    provider_class: Optional[Type[BaseProvider]] = None,
    **options: Any,
) -> None:
    """注册自定义厂商。

    兼容 OpenAI 协议的厂商::

        register_provider("myllm", base_url="https://llm.example.com/v1", env_key="MYLLM_KEY",
                          default_model="my-model")

    兼容 Anthropic 协议的端点::

        register_provider("my-claude-proxy", base_url="https://proxy/v1", api_format="anthropic")

    私有协议::

        register_provider("private", base_url="https://...", provider_class=MyProvider)
    """
    if provider_class is not None:
        api_format = f"_cls_{name.lower()}"
        FORMATS[api_format] = provider_class
    if api_format.lower() not in FORMATS:
        raise ValueError(f"未知 api_format: {api_format}，可选: {sorted(FORMATS)}")
    PROVIDERS[name.lower()] = dict(api_format=api_format.lower(), base_url=base_url, env_key=env_key,
                                   default_model=default_model, headers=headers, **options)


def list_providers():
    return sorted(PROVIDERS)


def _env(env_key: Any) -> Optional[str]:
    if not env_key:
        return None
    for k in (env_key if isinstance(env_key, (list, tuple)) else [env_key]):
        if os.getenv(k):
            return os.getenv(k)
    return None


def create_provider(
    provider: Any = "openai",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    default_model: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    api_format: Optional[str] = None,
    **options: Any,
) -> BaseProvider:
    if isinstance(provider, BaseProvider):
        return provider
    if isinstance(provider, type) and issubclass(provider, BaseProvider):
        return provider(api_key=api_key, base_url=base_url, default_model=default_model, headers=headers, **options)

    name = str(provider).lower()
    name = ALIASES.get(name, name)
    preset = dict(PROVIDERS.get(name) or {})
    if not preset:
        if not base_url:
            raise ValueError(f"未知厂商 '{provider}'。请先 register_provider()，或同时传入 base_url（默认按 OpenAI 协议）")
        preset = {"api_format": api_format or "openai"}
    fmt = (api_format or preset.pop("api_format", "openai")).lower()
    preset.pop("api_format", None)
    cls = FORMATS.get(fmt)
    if cls is None:
        raise ValueError(f"未知 api_format: {fmt}")
    env_key = preset.pop("env_key", None)
    merged_headers = dict(preset.pop("headers", None) or {})
    merged_headers.update(headers or {})
    preset_url = preset.pop("base_url", None)
    preset_model = preset.pop("default_model", None)
    preset.update(options)
    return cls(
        api_key=api_key or _env(env_key),
        base_url=base_url or preset_url,
        default_model=default_model or preset_model,
        headers=merged_headers,
        name=name,
        **preset,
    )
