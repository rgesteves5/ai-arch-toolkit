"""Meta provider — Muse Spark over the Meta Model API's Responses surface.

Meta ships no SDK of its own: its documentation points OpenAI-format clients at
``https://api.meta.ai/v1``. This adapter drives the official ``openai`` SDK's Responses API, the
surface Meta recommends for agents because it is the only OpenAI-compatible one that carries the
model's reasoning across tool turns (Chat Completions redacts it).

What belongs to the Responses API (input items and the replay of ``_raw``, tools, assembly,
streams, failures) lives in ``_responses.py``. This module holds what is Meta's: the host and the
client, the efforts, the error codes, ``tool_choice`` only ``auto``, function tools without
``strict``, and the hosted ``web_search``.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

from ai_arch_toolkit.core._exceptions import RequestError
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._model_id import lookup
from ai_arch_toolkit.core._providers._base import Options, Prepared, on_request, parse_options
from ai_arch_toolkit.core._providers._imports import require_sdk
from ai_arch_toolkit.core._response import Response

with require_sdk("meta"):
    import openai
    from openai.types.responses import Response as SDKResponse
    from openai.types.responses import ResponseInputItemParam
    from openai.types.shared.reasoning_effort import ReasoningEffort
    from openai.types.shared_params import Reasoning

    from ai_arch_toolkit.core._providers._responses import (
        Params,
        ResponsesProfile,
        ResponsesProvider,
        _parse_sdk_response,
        input_items,
        request_params,
    )

DEFAULT_BASE_URL = "https://api.meta.ai/v1"

# Parameters forwarded to ``responses.create()`` as given (``max_tokens`` is renamed).
_FORWARDED = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "parallel_tool_calls",
        "prompt_cache_key",
        "safety_identifier",
    }
)

# Muse Spark's reasoning efforts (https://dev.meta.ai/docs/reasoning): "max" only on the standard
# tier of muse-spark-1.3, not on contributor-tier or older models; a newer model gets them all.
# "none" is not among them: Muse Spark always reasons and answers it with a 400.
_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh", "max"})
_MODEL_EFFORTS: dict[str, frozenset[str]] = dict.fromkeys(
    (
        "muse-spark-1.3-contributor",
        "muse-spark-1.2",
        "muse-spark-1.2-contributor",
        "muse-spark-1.1",
    ),
    _EFFORTS - {"max"},
)

# Meta's error codes and their HTTP statuses (https://dev.meta.ai/docs/error-handling). A failure
# inside a response or a stream carries only its code; "server_error" is the Responses API's code
# for the failure Meta answers with a 500 of type server_error. A 400 and a 500 both come with no
# code: an unknown or missing code gets no status.
_CODE_STATUS: dict[str, int] = {
    "invalid_api_key": 401,
    "billing_not_configured": 402,
    "model_not_found": 404,
    "file_not_found": 404,
    "payload_too_large": 413,
    "rate_limit_exceeded": 429,
    "server_error": 500,
    "server_shutting_down": 503,
    "service_overloaded": 503,
    "backend_unavailable": 503,
    "gateway_timeout": 504,
}

# Meta serves one family of models, Muse Spark: only muse-spark- ids are routed here (D14).
_MUSE_SPARK = "muse-spark"

_PROFILE = ResponsesProfile(
    provider="Meta",
    families={"muse-spark-": _MUSE_SPARK},
    # Stateless requests get the encrypted reasoning back only when they ask for it (D12).
    include=("reasoning.encrypted_content",),
    # Only "auto" exists: "none" is sent as no tools, a forced choice is refused (D13).
    takes_tool_choice=False,
    # Meta constrains decoding either way; strict: true refuses plain Pydantic schemas (D13).
    function_strict=None,
    hosted_tools={"web_search": {"type": "web_search"}},
    codes=_CODE_STATUS,
)


def _input_items(messages: list[dict[str, Any]]) -> list[ResponseInputItemParam]:
    """Responses input items for a Meta request.

    Meta serves one family, Muse Spark, so a turn's reasoning goes back when a Muse Spark model
    produced it, whichever model the request names.
    """
    return input_items(messages, _PROFILE, _MUSE_SPARK)


def _http() -> Any:
    """The SDK's HTTP client, with the hook that marks a request as handed to the transport."""
    return openai.DefaultAsyncHttpxClient(event_hooks={"request": [on_request]})


class MetaProvider(ResponsesProvider):
    """Meta Model API provider (Muse Spark) via the ``openai`` SDK's Responses API."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        # Retry ownership belongs to LLM(RetryConfig(...)): hidden SDK retries
        # would be neither metered nor represented in Response.attempts.
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url or DEFAULT_BASE_URL,
            "max_retries": 0,
        }
        if timeout is not None:
            client_kwargs["timeout"] = timeout

        def _new_client() -> openai.AsyncOpenAI:
            client = openai.AsyncOpenAI(**client_kwargs, http_client=_http())
            # The SDK reads OPENAI_ORG_ID / OPENAI_PROJECT_ID from the environment and sends them
            # as headers; they identify an OpenAI account and must not reach Meta.
            client.organization = None
            client.project = None
            return client

        super().__init__(model, _PROFILE, _new_client)

    # ------------------------------------------------------------------
    # The contract (the I/O, the usage and the errors are the Responses core's)
    # ------------------------------------------------------------------

    def prepare(self, request: Request) -> Prepared[Params]:
        options = parse_options(request.kwargs, _FORWARDED, "Meta")
        items = _input_items(request.messages)
        reasoning = self._reasoning(options)
        params = request_params(
            self._profile,
            self._model,
            items,
            system=request.system,
            tools=request.tools,
            options=options,
        )
        if reasoning:
            params["reasoning"] = reasoning
        return Prepared(params, output_schema=options.output_schema)

    def _reasoning(self, options: Options) -> Reasoning:
        """Muse Spark always reasons (D13): the effort applies on its own, and thinking asks for
        a summary. Meta rejects logprobs from a reasoning model (https://dev.meta.ai/docs/reasoning).
        """
        if options.logprobs:
            raise RequestError(f"{self._model} reasons: Meta returns no logprobs")
        if options.thinking_budget:
            warnings.warn(
                "thinking_budget is not supported by Meta (Muse Spark takes thinking_effort), "
                "ignoring",
                stacklevel=5,
            )
        reasoning: Reasoning = {}
        effort = options.thinking_effort
        if effort is not None:
            found = lookup(self._model, _MODEL_EFFORTS)
            efforts = found.value if found is not None else _EFFORTS
            if effort not in efforts:
                raise RequestError(
                    f"{self._model} takes thinking_effort in {sorted(efforts)}, not {effort!r}"
                )
            reasoning["effort"] = cast("ReasoningEffort", effort)
        if options.thinking:
            reasoning["summary"] = "auto"
        return reasoning

    def assemble(self, final: SDKResponse, prepared: Prepared[Params]) -> Response:
        return _parse_sdk_response(final, self._model, output_schema=prepared.output_schema)

    async def count_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        """Count input tokens with Meta's ``POST /v1/responses/input_tokens``."""
        return await self._count_input_tokens(_input_items(messages), system, tools)
