"""Provider registry — maps model strings to provider instances."""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from ai_arch_toolkit.core._providers._base import BaseProvider

__all__ = ["create_provider"]

logger = logging.getLogger(__name__)

_MODEL_PREFIXES: dict[str, str] = {
    "claude-": "anthropic",
    "gpt-": "openai",
    "o1-": "openai",
    "o3-": "openai",
    "o4-": "openai",
    "grok-": "xai",
    "gemini-": "gemini",
}

_MODEL_IDS: dict[str, str] = {
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
}

# Derived from the routing table so it can never drift from what create_provider builds.
_PROVIDER_NAMES: frozenset[str] = frozenset(_MODEL_PREFIXES.values())

# Loopback hosts run on the user's own machine — no auth, key is optional.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})

# Local OpenAI-compatible servers (Ollama, LM Studio, vLLM) ignore Authorization.
_PLACEHOLDER_KEY = "not-needed"


def _match_provider(model: str) -> str | None:
    """Map a model string to a provider name via prefix matching, or None."""
    if provider := _MODEL_IDS.get(model):
        return provider
    for prefix, provider in _MODEL_PREFIXES.items():
        if model.startswith(prefix):
            return provider
    return None


def _unknown_model_error(model: str) -> ValueError:
    return ValueError(
        f"Cannot detect provider for model {model!r}. "
        f"Known model IDs: {sorted(_MODEL_IDS)}. Known prefixes: {sorted(_MODEL_PREFIXES)}. "
        "For local or OpenAI-compatible servers, pass provider='openai' and/or base_url=."
    )


def _detect_provider(model: str) -> str:
    """Map a model string to a provider name via prefix matching."""
    if provider := _match_provider(model):
        return provider
    raise _unknown_model_error(model)


def _is_local_url(base_url: str | None) -> bool:
    """Return True when base_url points at a loopback host (local server)."""
    if not base_url:
        return False
    host = urlsplit(base_url).hostname
    if not host:
        return False
    return host in _LOOPBACK_HOSTS or host.startswith("127.") or host.endswith(".localhost")


def _env_var_names(env_var: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(env_var, str):
        return (env_var,)
    return env_var


def _format_env_var_names(names: tuple[str, ...]) -> str:
    if len(names) == 1:
        return f"the {names[0]} environment variable"
    return "one of the " + ", ".join(names[:-1]) + f", or {names[-1]} environment variables"


def _resolve_key(
    env_var: str | tuple[str, ...], api_key: str | None, *, local: bool = False
) -> str:
    """Resolve an API key, preferring an explicit one, then env vars in order.

    For a local (loopback) server the env var is **not** consulted — a real
    cloud key is never sent to localhost — and a placeholder is used when no
    key is passed explicitly. Remote endpoints still require a key.
    """
    if api_key:
        return api_key
    if local:
        return _PLACEHOLDER_KEY
    names = _env_var_names(env_var)
    for name in names:
        if key := os.environ.get(name, ""):
            return key
    raise ValueError(f"No API key provided. Pass api_key= or set {_format_env_var_names(names)}.")


def create_provider(
    model: str,
    *,
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
) -> BaseProvider:
    """Create a provider instance from a model string.

    Routing precedence: an explicit ``provider`` wins; otherwise the model
    prefix is matched (``claude-`` → Anthropic, ``gpt-``/``o1-`` → OpenAI,
    ``grok-`` → xAI, ``gemini-`` → Gemini); otherwise an unknown model with
    ``base_url`` set falls back to the OpenAI-compatible adapter (Ollama, LM
    Studio, vLLM). The API key is required unless ``base_url`` points at a
    loopback host (localhost), where local servers ignore it.

    Args:
        provider: Force a specific provider, bypassing prefix detection.
            One of ``anthropic``, ``openai``, ``xai``, ``gemini``.
        base_url: Override the endpoint. Only the Anthropic and OpenAI
            adapters honor it; xAI and Gemini ignore it with a warning.
    """
    base_url = base_url or None  # normalize "" so it never masks the key check
    local = _is_local_url(base_url)

    if provider is not None:
        if provider not in _PROVIDER_NAMES:
            raise ValueError(
                f"Unknown provider {provider!r}. Valid providers: {sorted(_PROVIDER_NAMES)}"
            )
        name = provider
    elif matched := _match_provider(model):
        name = matched
    elif base_url:
        logger.debug(
            "Unknown model %r with base_url set; routing to OpenAI-compatible provider", model
        )
        name = "openai"
    else:
        raise _unknown_model_error(model)

    if name == "anthropic":
        from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider

        return AnthropicProvider(
            model,
            _resolve_key("ANTHROPIC_API_KEY", api_key, local=local),
            base_url=base_url,
            timeout=timeout,
        )

    if name == "openai":
        from ai_arch_toolkit.core._providers._openai import OpenAIProvider

        return OpenAIProvider(
            model,
            _resolve_key("OPENAI_API_KEY", api_key, local=local),
            base_url=base_url,
            timeout=timeout,
        )

    if name == "xai":
        from ai_arch_toolkit.core._providers._xai import XAIProvider

        if base_url:
            warnings.warn(f"base_url is not supported by {name} provider, ignoring", stacklevel=2)
        return XAIProvider(
            model,
            _resolve_key("XAI_API_KEY", api_key),
            timeout=timeout,
        )

    if name == "gemini":
        from ai_arch_toolkit.core._providers._gemini import GeminiProvider

        if base_url:
            warnings.warn(f"base_url is not supported by {name} provider, ignoring", stacklevel=2)
        return GeminiProvider(
            model,
            _resolve_key(("GOOGLE_API_KEY", "GEMINI_API_KEY"), api_key),
            timeout=timeout,
        )

    raise NotImplementedError(f"Provider {name!r} is not yet implemented.")
