"""Provider registry — maps model strings to provider instances."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_arch_toolkit.core._http import RetryConfig
    from ai_arch_toolkit.core._providers._base import BaseProvider

logger = logging.getLogger(__name__)

_MODEL_PREFIXES: dict[str, str] = {
    "claude-": "anthropic",
    "gpt-": "openai",
    "o1-": "openai",
    "o3-": "openai",
    "o4-": "openai",
}


def _detect_provider(model: str) -> str:
    """Map a model string to a provider name via prefix matching."""
    for prefix, provider in _MODEL_PREFIXES.items():
        if model.startswith(prefix):
            return provider
    raise ValueError(
        f"Cannot detect provider for model {model!r}. Known prefixes: {sorted(_MODEL_PREFIXES)}"
    )


def _resolve_key(env_var: str, api_key: str | None) -> str:
    key = api_key or os.environ.get(env_var, "")
    if not key:
        raise ValueError(
            f"No API key provided. Pass api_key= or set the {env_var} environment variable."
        )
    return key


def create_provider(
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    retry: RetryConfig | None = None,
) -> BaseProvider:
    """Create a provider instance from a model string."""
    name = _detect_provider(model)

    if name == "anthropic":
        from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider

        return AnthropicProvider(
            model,
            _resolve_key("ANTHROPIC_API_KEY", api_key),
            base_url=base_url,
            retry=retry,
        )

    if name == "openai":
        from ai_arch_toolkit.core._providers._openai import OpenAIProvider

        return OpenAIProvider(
            model,
            _resolve_key("OPENAI_API_KEY", api_key),
            base_url=base_url,
            retry=retry,
        )

    raise NotImplementedError(f"Provider {name!r} is not yet implemented.")
