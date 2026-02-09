"""Provider registry — maps provider names to factory functions."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ai_arch_toolkit.llm._providers._anthropic import AnthropicProvider
from ai_arch_toolkit.llm._providers._gemini import GeminiProvider
from ai_arch_toolkit.llm._providers._openai_compat import (
    OPENAI_COMPAT_PROVIDERS,
    OpenAICompatProvider,
)
from ai_arch_toolkit.llm._providers._openai_responses import OpenAIResponsesProvider
from ai_arch_toolkit.llm._providers._xai_responses import XAIResponsesProvider

if TYPE_CHECKING:
    from ai_arch_toolkit.llm._http import RetryConfig
    from ai_arch_toolkit.llm._providers._base import BaseProvider


def _resolve_key(env_var: str, api_key: str | None) -> str:
    key = api_key or os.environ.get(env_var, "")
    if not key:
        raise ValueError(
            f"No API key provided. Pass api_key= or set the {env_var} environment variable."
        )
    return key


def create_provider(
    name: str,
    model: str,
    api_key: str | None = None,
    *,
    retry: RetryConfig | None = None,
) -> BaseProvider:
    """Create a provider instance by name."""
    if name in OPENAI_COMPAT_PROVIDERS:
        env_var = OPENAI_COMPAT_PROVIDERS[name]["env_key"]
        return OpenAICompatProvider(name, model, _resolve_key(env_var, api_key), retry=retry)

    if name == "anthropic":
        return AnthropicProvider(model, _resolve_key("ANTHROPIC_API_KEY", api_key), retry=retry)

    if name == "gemini":
        return GeminiProvider(model, _resolve_key("GEMINI_API_KEY", api_key), retry=retry)

    if name == "openai-responses":
        return OpenAIResponsesProvider(model, _resolve_key("OPENAI_API_KEY", api_key), retry=retry)

    if name == "xai-responses":
        return XAIResponsesProvider(model, _resolve_key("XAI_API_KEY", api_key), retry=retry)

    supported = sorted(
        [
            *OPENAI_COMPAT_PROVIDERS,
            "anthropic",
            "gemini",
            "openai-responses",
            "xai-responses",
        ]
    )
    raise ValueError(f"Unknown provider {name!r}. Supported: {supported}")
