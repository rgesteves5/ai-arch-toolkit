"""Prototype of the reusable helper: ``validate_call(provider, method, kwargs) -> list[str]``.

One registry maps each SDK entry point an adapter calls to the SDK's own request contract:
Stainless TypedDict params (anthropic/openai) via ``strict.strict_validator``; google-genai via the
SDK's offline request pipeline; xai-sdk by replaying into the real ``chat.create()`` (protobuf).
"""

from __future__ import annotations

import functools
import importlib
import inspect
import warnings
from typing import Any

from strict import check, strict_validator

# (module, non-streaming type, streaming type | None)
_TYPED: dict[str, dict[str, tuple[str, str, str | None]]] = {
    "AnthropicProvider": {
        "messages.create": ("anthropic.types.message_create_params", "MessageCreateParamsNonStreaming", "MessageCreateParamsStreaming"),
        "messages.stream": ("anthropic.types.message_create_params", "MessageCreateParamsNonStreaming", None),
        "messages.count_tokens": ("anthropic.types.message_count_tokens_params", "MessageCountTokensParams", None),
        "messages.batches.create": ("anthropic.types.messages.batch_create_params", "BatchCreateParams", None),
    },
    "OpenAIProvider": {
        "chat.completions.create": ("openai.types.chat.completion_create_params", "CompletionCreateParamsNonStreaming", "CompletionCreateParamsStreaming"),
        "batches.create": ("openai.types.batch_create_params", "BatchCreateParams", None),
    },
    "MetaProvider": {
        "responses.create": ("openai.types.responses.response_create_params", "ResponseCreateParamsNonStreaming", "ResponseCreateParamsStreaming"),
        "responses.input_tokens.count": ("openai.types.responses.input_token_count_params", "InputTokenCountParams", None),
    },
}
ENTRY_POINTS: dict[str, tuple[str, ...]] = {
    **{name: tuple(methods) for name, methods in _TYPED.items()},
    "GeminiProvider": ("aio.models.generate_content", "aio.models.generate_content_stream"),
    "XAIProvider": ("chat.create",),
}
# Per-request options of every Stainless method; not part of the params TypedDict.
_REQUEST_OPTIONS = frozenset({"extra_headers", "extra_query", "extra_body", "timeout"})


@functools.cache
def _validator(module: str, name: str) -> Any:
    return strict_validator(getattr(importlib.import_module(module), name))


def _is_sentinel(value: Any) -> bool:
    return type(value).__name__ in ("NotGiven", "Omit")


def _validate_typed(provider: str, method: str, kwargs: dict[str, Any]) -> list[str]:
    module, plain, streaming = _TYPED[provider][method]
    name = streaming if (streaming and kwargs.get("stream") is True) else plain
    payload = {k: v for k, v in kwargs.items() if k not in _REQUEST_OPTIONS and not _is_sentinel(v)}
    return check(_validator(module, name), payload)


@functools.cache
def _gemini() -> tuple[Any, Any, Any, dict[str, inspect.Signature]]:
    from google import genai
    from google.genai import models, types

    sigs = {
        "aio.models.generate_content": inspect.signature(models.AsyncModels.generate_content),
        "aio.models.generate_content_stream": inspect.signature(models.AsyncModels.generate_content_stream),
    }
    return genai.Client(api_key="offline")._api_client, models, types, sigs


def _validate_gemini(method: str, kwargs: dict[str, Any]) -> list[str]:
    api_client, models, types, sigs = _gemini()
    try:
        sigs[method].bind(None, **kwargs)
    except TypeError as exc:
        return [f"signature | {exc}"]
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*is not a valid.*")  # lenient SDK enums
        try:
            params = types._GenerateContentParameters(**kwargs)
            types._GenerateContentParameters.model_validate(params.model_dump())
            models._GenerateContentParameters_to_mldev(api_client, params)
        except Exception as exc:
            return [f"{type(exc).__name__} | {' '.join(str(exc).split())[:260]}"]
    return []


@functools.cache
def _xai() -> tuple[Any, inspect.Signature]:
    import xai_sdk
    from xai_sdk import chat as base_chat

    # The sync client: same ``create()`` (defined once on the shared base class), but building it
    # needs no running event loop (``AsyncClient()`` does, which breaks in a sync fixture).
    return xai_sdk.Client(api_key="offline"), inspect.signature(base_chat.BaseClient.create)


def _validate_xai(kwargs: dict[str, Any]) -> list[str]:
    client, sig = _xai()
    try:
        sig.bind(None, **kwargs)
        client.chat.create(**kwargs).proto.SerializeToString()
    except Exception as exc:
        return [f"{type(exc).__name__} | {str(exc)[:260]}"]
    return []


def validate_call(provider: str, method: str, kwargs: dict[str, Any]) -> list[str]:
    """Violations of the SDK's own request contract for one captured call (empty = conforms)."""
    if provider in _TYPED:
        return _validate_typed(provider, method, kwargs)
    if provider == "GeminiProvider":
        return _validate_gemini(method, kwargs)
    if provider == "XAIProvider":
        return _validate_xai(kwargs)
    raise KeyError(provider)
