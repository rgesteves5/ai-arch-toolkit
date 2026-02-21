"""Provider registry — maps provider names to factory functions."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import httpx
import requests

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

logger = logging.getLogger(__name__)


def _resolve_key(env_var: str, api_key: str | None) -> str:
    key = api_key or os.environ.get(env_var, "")
    if not key:
        logger.warning("Missing API key for env var %s", env_var)
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
    created: BaseProvider
    logger.debug("Creating provider name=%s model=%s", name, model)
    if name in OPENAI_COMPAT_PROVIDERS:
        env_var = OPENAI_COMPAT_PROVIDERS[name]["env_key"]
        created = OpenAICompatProvider(name, model, _resolve_key(env_var, api_key), retry=retry)
        created._session = requests.Session()
        created._async_client = httpx.AsyncClient()
        return created

    if name == "anthropic":
        created = AnthropicProvider(model, _resolve_key("ANTHROPIC_API_KEY", api_key), retry=retry)
        created._session = requests.Session()
        created._async_client = httpx.AsyncClient()
        return created

    if name == "gemini":
        created = GeminiProvider(model, _resolve_key("GEMINI_API_KEY", api_key), retry=retry)
        created._session = requests.Session()
        created._async_client = httpx.AsyncClient()
        return created

    if name == "openai-responses":
        created = OpenAIResponsesProvider(
            model, _resolve_key("OPENAI_API_KEY", api_key), retry=retry
        )
        created._session = requests.Session()
        created._async_client = httpx.AsyncClient()
        return created

    if name == "xai-responses":
        created = XAIResponsesProvider(model, _resolve_key("XAI_API_KEY", api_key), retry=retry)
        created._session = requests.Session()
        created._async_client = httpx.AsyncClient()
        return created

    supported = sorted(
        [
            *OPENAI_COMPAT_PROVIDERS,
            "anthropic",
            "gemini",
            "openai-responses",
            "xai-responses",
        ]
    )
    logger.warning("Unknown provider requested: %s", name)
    raise ValueError(f"Unknown provider {name!r}. Supported: {supported}")
