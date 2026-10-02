"""OpenAI provider — OpenAI's own host through the Responses API, on the shared core (D43).

From GPT-5.4 on, Chat Completions takes tool calls only at the ``none`` effort ("Starting with
GPT-5.4, Chat Completions does not support tool calling with ``reasoning_effort`` values other
than ``none``", https://developers.openai.com/api/docs/guides/migrate-to-responses), and only the
Responses API returns reasoning summaries and the encrypted reasoning a tool loop replays. So
OpenAI's own host (no ``base_url``, or one on ``api.openai.com``) is served here, and any other
host is an OpenAI-compatible server on Chat Completions (``_openai_compatible.py``):
``create_provider`` chooses by host.

What belongs to the Responses API (input items and the replay of ``_raw``, tools, assembly,
streams, failures) lives in ``_responses.py``. This module holds what is OpenAI's: the profile
(model families, error codes, ``strict: false`` function tools, the hosted ``web_search``), each
model's rules, the Chat Completions parameters the Responses API has no place for, token counting
and the Batch API on ``/v1/responses``.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast, get_args

from ai_arch_toolkit.core._exceptions import RequestError
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._model_id import lookup
from ai_arch_toolkit.core._providers._base import Options, Prepared, parse_options
from ai_arch_toolkit.core._providers._imports import require_sdk
from ai_arch_toolkit.core._response import Response

with require_sdk("openai"):
    import openai
    from openai.types.responses import Response as SDKResponse
    from openai.types.responses import ResponseIncludable, ResponseInputItemParam
    from openai.types.shared.reasoning_effort import ReasoningEffort
    from openai.types.shared_params import Reasoning

    from ai_arch_toolkit.core._providers._openai_compatible import (
        CHAT_COMPLETIONS,
        batch_output,
        batch_request,
        batch_results,
        batch_status,
        chat_batch_response,
        submit_batch,
    )
    from ai_arch_toolkit.core._providers._responses import (
        Params,
        ResponsesProfile,
        ResponsesProvider,
        _parse_sdk_response,
        http_client,
        input_items,
        request_params,
    )

# Parameters forwarded to ``responses.create()`` as given (the output limit is renamed).
_FORWARDED = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "parallel_tool_calls",
        "top_logprobs",
    }
)

# Chat Completions parameters the Responses API has no place for, refused before sending (D43,
# D44): live, stop and seed get a 400 unknown_parameter and the two penalties a 500 after ~90 s,
# which the LLM's retries would repeat (O01, 2026-10-02). Structured output goes through
# output_schema or json_mode. An OpenAI-compatible server (another host) still takes all five.
_CHAT_COMPLETIONS_ONLY = frozenset(
    {"stop", "seed", "frequency_penalty", "presence_penalty", "response_format"}
)

# Sampling parameters a GPT-6 model drops while it reasons.
_SAMPLING = ("temperature", "top_p", "top_logprobs")
_LOGPROBS: ResponseIncludable = "message.output_text.logprobs"
_EFFORTS = frozenset(get_args(get_args(ReasoningEffort)[0]))


@dataclass(frozen=True, slots=True, kw_only=True)
class _Model:
    """What one family of OpenAI models takes through the Responses API.

    Attributes:
        efforts: The reasoning efforts it takes; none for a model that does not reason, which
            refuses ``thinking``.
        default_effort: The effort it reasons at when the request sends none, where its page
            gives one; a model that takes no ``"none"`` always reasons.
        sampling_while_reasoning: It takes ``temperature``, ``top_p`` and logprobs while it
            reasons. The GPT-6 models do not: "When reasoning effort is not ``none``, remove
            ``temperature``, ``top_p``, and ``top_logprobs``"
            (https://developers.openai.com/api/docs/guides/latest-model; live, O01 check 7).
    """

    efforts: frozenset[str] = _EFFORTS
    default_effort: str | None = None
    sampling_while_reasoning: bool = True

    def reasons_at(self, effort: str | None) -> bool:
        """Whether a request at ``effort`` (``None``: the request sends none) reasons."""
        if not self.efforts:
            return False
        effort = effort or self.default_effort
        return "none" not in self.efforts if effort is None else effort != "none"


_CURRENT = _Model()  # GPT-5 to GPT-5.6, o3, and a model not listed (a new one)
# The families before the reasoning generation: https://developers.openai.com/api/docs/models/gpt-4o
_LEGACY = _Model(efforts=frozenset())
# The GPT-6 models take "max", which their pages list ("reasoning.effort supports low, medium,
# high, xhigh, and max", https://developers.openai.com/api/docs/models/gpt-6-astra) and the
# Responses API takes (live on gpt-6.1-sol, O01 check 8; Chat Completions refused it).
_REASONING = frozenset({"low", "medium", "high", "xhigh", "max"})
# No "none" (nor "minimal"), so Astra always reasons
# (https://developers.openai.com/api/docs/models/gpt-6-astra).
_ASTRA = _Model(efforts=_REASONING, sampling_while_reasoning=False)
# GPT-6.1 Sol takes no "none" (nor "minimal") either and reasons at "medium" by default
# (https://developers.openai.com/api/docs/models/gpt-6.1-sol).
_SOL_6_1 = _Model(efforts=_REASONING, default_effort="medium", sampling_while_reasoning=False)
# Sol and Luna take "none" and reason at "medium" when the request sends no effort
# (https://developers.openai.com/api/docs/models/gpt-6-sol and .../gpt-6-luna).
_SOL_LUNA = _Model(
    efforts=_REASONING | {"none"}, default_effort="medium", sampling_while_reasoning=False
)

# The families that differ from the current generation, a closed list: a model not listed gets the
# current generation's rules. https://developers.openai.com/api/docs/models
_MODELS: dict[str, _Model] = {
    "gpt-6-astra": _ASTRA,
    "gpt-6.1-sol": _SOL_6_1,
    **dict.fromkeys(("gpt-6-sol", "gpt-6-luna"), _SOL_LUNA),
    **dict.fromkeys(("gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"), _LEGACY),
}

# Persisted reasoning "can be reused only within the same model family. For example,
# gpt-5.6-sol, gpt-5.6-terra, and gpt-5.6-luna can reuse each other's reasoning, but reasoning
# does not carry between the GPT-5.6 and GPT-5.5 families"
# (https://developers.openai.com/api/docs/guides/reasoning). A family is a generation: its tiers
# and snapshots share it (gpt-5.6-terra and gpt-5.6-luna are gpt-5.6), its own id is the family by
# itself, and an id no generation claims (o3, gpt-4o) is its own family. OpenAI takes another
# family's reasoning without an error (O01 check 4): the guard saves the payload.
_GENERATIONS = (
    "gpt-6.1",
    "gpt-6",
    "gpt-5.6",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.3",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
)

# The statuses of the codes a failure inside a response or a stream carries (ResponseError.code
# in the SDK): the two the error guide names, "429 Rate limit reached" and "500 Server error"
# (https://developers.openai.com/api/docs/guides/error-codes). The others (an invalid prompt or
# image, a policy) are documented with no status, and stay a ResponseError.
_CODE_STATUS: dict[str, int] = {"rate_limit_exceeded": 429, "server_error": 500}

_PROFILE = ResponsesProfile(
    provider="OpenAI",
    families={f"{generation}-": generation for generation in _GENERATIONS},
    # With store: false the reasoning items carry their encrypted content without asking
    # (https://developers.openai.com/api/docs/guides/reasoning; live, O01 check 2).
    include=(),
    takes_tool_choice=True,
    # "In Responses, omitting strict attempts strict mode" (migration guide): live, OpenAI then
    # made an optional parameter required (O01 check 5). Sent as the schema was written.
    function_strict=False,
    output_strict=True,
    hosted_tools={"web_search": {"type": "web_search"}},  # D44; a config belongs to C05
    codes=_CODE_STATUS,
)


def _refuse_chat_completions_only(kwargs: Mapping[str, object]) -> None:
    if found := sorted(_CHAT_COMPLETIONS_ONLY & kwargs.keys()):
        raise RequestError(
            f"OpenAI's Responses API takes no {', '.join(found)} (structured output is "
            "output_schema= or json_mode=); an OpenAI-compatible server on another base_url "
            "still takes them"
        )


def _sampling(params: Params, options: Options, model: _Model, effort: str | None) -> None:
    """The sampling parameters and logprobs the model takes at the request's effort.

    The ``LLM`` always sends a ``temperature``, so what the model does not take is dropped
    rather than refused. Logprobs come with their ``include`` and the ``top_logprobs`` given
    (live, O01 check 7).
    """
    reasons = model.reasons_at(effort)
    if reasons and not model.sampling_while_reasoning:
        for name in _SAMPLING:
            params.pop(name, None)
        return
    if reasons and params.get("temperature") not in (None, 1, 1.0):
        # The reasoning models before GPT-6 take only the default temperature while they reason
        # (live probe of 2026-04-28 in scripts/model_probe_notes.md).
        params.pop("temperature")
    if options.logprobs:
        params["include"] = [*(params.get("include") or ()), _LOGPROBS]


class OpenAIProvider(ResponsesProvider):
    """OpenAI's own API via the ``openai`` SDK's Responses API."""

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
        client_kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
        if base_url:
            client_kwargs["base_url"] = base_url
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        super().__init__(
            model,
            _PROFILE,
            lambda: openai.AsyncOpenAI(**client_kwargs, http_client=http_client()),
        )

    def _rules(self) -> _Model:
        found = lookup(self._model, _MODELS)
        return found.value if found is not None else _CURRENT

    def _input_items(self, messages: list[dict[str, Any]]) -> list[ResponseInputItemParam]:
        """Responses input items, replaying the reasoning of this model's family only."""
        return input_items(messages, self._profile, self._profile.family(self._model))

    # ------------------------------------------------------------------
    # The contract (the I/O, the usage and the errors are the Responses core's)
    # ------------------------------------------------------------------

    def prepare(self, request: Request) -> Prepared[Params]:
        _refuse_chat_completions_only(request.kwargs)
        options = parse_options(request.kwargs, _FORWARDED, "OpenAI")
        model = self._rules()
        reasoning = self._reasoning(options, model)
        params = request_params(
            self._profile,
            self._model,
            self._input_items(request.messages),
            system=request.system,
            tools=request.tools,
            options=options,
        )
        if reasoning:
            params["reasoning"] = reasoning
        _sampling(params, options, model, reasoning.get("effort"))
        return Prepared(params, output_schema=options.output_schema)

    def _reasoning(self, options: Options, model: _Model) -> Reasoning:
        """With ``thinking``: the effort (the one given, else ``"high"``) among the model's, and
        a summary of the reasoning, the source of the thinking blocks
        (https://developers.openai.com/api/docs/guides/reasoning). Without it the model reasons
        at its own default."""
        if options.thinking_budget:
            warnings.warn(
                "thinking_budget is not supported by OpenAI (it takes thinking_effort), ignoring",
                stacklevel=5,
            )
        if not options.thinking:
            return {}
        if not model.efforts:
            raise RequestError(f"{self._model} does not reason: thinking is not available")
        effort = options.thinking_effort or "high"
        if effort not in model.efforts:
            raise RequestError(
                f"{self._model} takes thinking_effort in {sorted(model.efforts)}, not {effort!r}"
            )
        reasoning: Reasoning = {"effort": cast("ReasoningEffort", effort)}
        if effort != "none":
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
        """Count input tokens with OpenAI's ``POST /v1/responses/input_tokens``."""
        return await self._count_input_tokens(self._input_items(messages), system, tools)

    # ------------------------------------------------------------------
    # batch
    # ------------------------------------------------------------------

    async def batch_submit(self, requests: list[dict[str, Any]], **kwargs: Any) -> str:
        """Submit a batch on ``/v1/responses`` (https://developers.openai.com/api/docs/guides/batch);
        each body is what ``prepare`` builds."""
        bodies = [
            (req.get("custom_id", ""), self.prepare(batch_request(req, self._model)).params)
            for req in requests
        ]
        with self._mapped():
            return await submit_batch(self._client, "/v1/responses", bodies)

    async def batch_status(self, batch_id: str) -> str:
        """Check batch status."""
        with self._mapped():
            return await batch_status(self._client, batch_id)

    async def batch_results(self, batch_id: str) -> list[Any]:
        """Retrieve completed batch results, read by the batch's endpoint: a batch submitted to
        Chat Completions before the official host moved to the Responses API still reads."""
        with self._mapped():
            endpoint, entries = await batch_output(self._client, batch_id)
        if endpoint == CHAT_COMPLETIONS:
            return batch_results(entries, lambda body: chat_batch_response(body, self._model))
        return batch_results(entries, self._batch_response)

    def _batch_response(self, body: dict[str, Any]) -> Response:
        """A Responses body from a batch's output, assembled as a call's, usage and cost
        included."""
        unrequested: Prepared[Params] = Prepared({})
        return self._answer(SDKResponse.model_construct(**body), unrequested).response
