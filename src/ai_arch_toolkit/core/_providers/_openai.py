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
the Batch API on ``/v1/responses``, and the GPT Image models' rules on the Images API
(``_openai_images.py``).
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast, get_args

from ai_arch_toolkit.core._exceptions import RequestError
from ai_arch_toolkit.core._images import ImageRequest
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._model_id import lookup
from ai_arch_toolkit.core._pricing import _estimate_response_cost
from ai_arch_toolkit.core._providers import OWN_BASE_URLS
from ai_arch_toolkit.core._providers._base import (
    DRAWS_ONLY,
    AdapterFacts,
    Options,
    Prepared,
    ThinkingMode,
    ordered_efforts,
    parse_options,
)
from ai_arch_toolkit.core._providers._imports import require_sdk
from ai_arch_toolkit.core._response import Response, Usage

with require_sdk("openai"):
    import openai
    from openai.types.responses import Response as SDKResponse
    from openai.types.responses import ResponseIncludable, ResponseInputItemParam
    from openai.types.responses.tool_param import ImageGeneration
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
    from ai_arch_toolkit.core._providers._openai_images import (
        ImageModel,
        ImagesCall,
        SizeTokens,
        TileTokens,
        image_size,
        image_token_bound,
        images_call,
    )
    from ai_arch_toolkit.core._providers._responses import (
        Call,
        Final,
        Params,
        ResponsesProfile,
        ResponsesProvider,
        _extract_usage,
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

    def thinking_mode(self) -> ThinkingMode:
        """``"none"`` without efforts, ``"always"`` when a request without one reasons."""
        if not self.efforts:
            return "none"
        return "always" if self.reasons_at(None) else "optional"


# Each model's efforts, and the effort a request without one runs at, were measured live on the
# Responses API on 2026-10-02 (O04): one request per effort and model, and the effort the response
# echoes. The pages list more than some models take (the GPT-5.5 refuses "max"), and a model that
# reasons by default refuses a temperature other than its default.
_REASONING = frozenset({"low", "medium", "high", "xhigh", "max"})
# A model not listed (a new one) gets the current generation's rules: every effort, and it reasons
# at "medium" when the request sends none, as GPT-5.5, GPT-5.6 and GPT-6 do.
_CURRENT = _Model(default_effort="medium")
# The families before the reasoning generation: https://developers.openai.com/api/docs/models/gpt-4o
_LEGACY = _Model(efforts=frozenset())
_GPT_5 = _Model(efforts=frozenset({"minimal", "low", "medium", "high"}), default_effort="medium")
_GPT_5_1 = _Model(efforts=frozenset({"none", "low", "medium", "high"}), default_effort="none")
_GPT_5_2 = _Model(efforts=_GPT_5_1.efforts | {"xhigh"}, default_effort="none")
_GPT_5_5 = _Model(efforts=_GPT_5_2.efforts, default_effort="medium")
_GPT_5_6 = _Model(efforts=_REASONING | {"none"}, default_effort="medium")
_O3 = _Model(efforts=frozenset({"low", "medium", "high"}), default_effort="medium")
# The pro models always reason, from "medium" up (GPT-5 pro only at "high").
_PRO = _Model(efforts=frozenset({"medium", "high", "xhigh"}), default_effort="medium")
_GPT_5_5_PRO = _Model(efforts=_PRO.efforts, default_effort="high")
_GPT_5_PRO = _Model(efforts=frozenset({"high"}), default_effort="high")
# No "none" (nor "minimal"), so Astra and GPT-6.1 Sol always reason
# (https://developers.openai.com/api/docs/models/gpt-6-astra, .../gpt-6.1-sol); "max" is on their
# pages, and the Responses API takes it (Chat Completions refused it, O01 check 8).
_ASTRA = _Model(efforts=_REASONING, default_effort="medium", sampling_while_reasoning=False)
_SOL_6_1 = _ASTRA
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
    **dict.fromkeys(("gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"), _GPT_5_6),
    "gpt-5.5": _GPT_5_5,
    **dict.fromkeys(("gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.2"), _GPT_5_2),
    "gpt-5.3-codex": _GPT_5_2,
    **dict.fromkeys(("gpt-5.4-pro", "gpt-5.2-pro"), _PRO),
    "gpt-5.5-pro": _GPT_5_5_PRO,
    "gpt-5-pro": _GPT_5_PRO,
    "gpt-5.1": _GPT_5_1,
    **dict.fromkeys(("gpt-5", "gpt-5-mini", "gpt-5-nano"), _GPT_5),
    "o3": _O3,
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

# The GPT Image models, on the Images API (https://developers.openai.com/api/reference/resources/
# images). gpt-image-2 and later take any size within limits, the earlier ones three fixed sizes
# (live, I01 check 5); "xhigh" and "max" only the 2.5 models. A dated snapshot takes its model's
# rules, and an unlisted gpt-image- id the newest ones.
#
# Each model's image output tokens per image (G-40), from the same guide: the 2.5 models and
# gpt-image-2 by its calculator's tiles per quality; the older ones by its table of "models prior
# to gpt-image-2" (1024x1024, 1024x1536, 1536x1024). gpt-image-1-mini has no table of its own: its
# counts are its per-image prices (https://developers.openai.com/api/docs/models/gpt-image-1-mini)
# at its $8 per 1M image output tokens, rounded up within the prices' precision.
_GPT_IMAGE_QUALITIES = frozenset({"low", "medium", "high", "auto"})
_SQUARE, _PORTRAIT, _LANDSCAPE = "1024x1024", "1024x1536", "1536x1024"


def _sized(*counts: tuple[str, int, int, int]) -> SizeTokens:
    return SizeTokens(
        {
            quality: {_SQUARE: sq, _PORTRAIT: tall, _LANDSCAPE: wide}
            for quality, sq, tall, wide in counts
        }
    )


_GPT_IMAGE_1_TOKENS = _sized(
    ("low", 272, 408, 400), ("medium", 1056, 1584, 1568), ("high", 4160, 6240, 6208)
)
_GPT_IMAGE_1_MINI_TOKENS = _sized(
    ("low", 687, 812, 812), ("medium", 1437, 1937, 1937), ("high", 4562, 6562, 6562)
)
_GPT_IMAGE_2_5 = ImageModel(
    size="free",
    qualities=_GPT_IMAGE_QUALITIES | {"xhigh", "max"},
    tokens=TileTokens({"low": 16, "medium": 24, "high": 48, "xhigh": 64, "max": 96}),
)
_IMAGE_MODELS: dict[str, ImageModel] = {
    **dict.fromkeys(("gpt-image-2.5-sunburst", "gpt-image-2.5-flare"), _GPT_IMAGE_2_5),
    "gpt-image-2": ImageModel(
        size="free",
        qualities=_GPT_IMAGE_QUALITIES,
        tokens=TileTokens({"low": 16, "medium": 48, "high": 96}),
    ),
    **dict.fromkeys(
        ("gpt-image-1.5", "chatgpt-image-latest", "gpt-image-1"),
        ImageModel(size="fixed", qualities=_GPT_IMAGE_QUALITIES, tokens=_GPT_IMAGE_1_TOKENS),
    ),
    "gpt-image-1-mini": ImageModel(
        size="fixed", qualities=_GPT_IMAGE_QUALITIES, tokens=_GPT_IMAGE_1_MINI_TOKENS
    ),
}
_IMAGE_FAMILIES = {"gpt-image-": _GPT_IMAGE_2_5}

# The statuses of the codes a failure inside a response or a stream carries (ResponseError.code
# in the SDK): the two the error guide names, "429 Rate limit reached" and "500 Server error"
# (https://developers.openai.com/api/docs/guides/error-codes). The others (an invalid prompt or
# image, a policy) are documented with no status, and stay a ResponseError.
_CODE_STATUS: dict[str, int] = {"rate_limit_exceeded": 429, "server_error": 500}


def _image_rules(model: str) -> ImageModel | None:
    found = lookup(model, _IMAGE_MODELS, _IMAGE_FAMILIES)
    return found.value if found is not None else None


def _rules_of(model: str) -> _Model:
    found = lookup(model, _MODELS)
    return found.value if found is not None else _CURRENT


def _image_tool(config: Mapping[str, Any]) -> ImageGeneration:
    """The hosted image generation tool for ``image_generation()``'s config, checked against its
    image model's rules (https://developers.openai.com/api/docs/guides/tools-image-generation)."""
    model = str(config.get("model", ""))
    rules = _image_rules(model)
    if rules is None:
        raise RequestError(f"image_generation: {model!r} is not a GPT Image model")
    options = ImageRequest(
        aspect_ratio=config.get("aspect_ratio"),
        resolution=config.get("resolution"),
        quality=config.get("quality"),
        output_format=config.get("output_format"),
    )
    if options.quality is not None and options.quality not in rules.qualities:
        raise RequestError(f"image_generation: {model} takes quality in {sorted(rules.qualities)}")
    tool: ImageGeneration = {"type": "image_generation", "model": model}
    if (size := image_size(options, rules.size)) is not None:
        tool["size"] = size
    if options.quality is not None:
        tool["quality"] = cast("Any", options.quality)
    if options.output_format is not None:
        tool["output_format"] = options.output_format
    if (partial := config.get("partial_images")) is not None:
        tool["partial_images"] = int(partial)
    return tool


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
    image_tool=_image_tool,
    image_replay="input",
)


def _hosted_image(final: SDKResponse) -> tuple[str, Usage] | None:
    """The image model and the usage of the hosted image tool, when the turn drew.

    OpenAI reports the tool's tokens in a top-level ``tool_usage.image_gen``, with the Images
    API's usage shape, apart from the turn's ``usage`` (live, I01 check 2); the SDK does not type
    it. The image model is the one the request's tool named, echoed in ``tools``.
    """
    reported = (final.model_extra or {}).get("tool_usage")
    image_gen = reported.get("image_gen") if isinstance(reported, dict) else None
    if not isinstance(image_gen, dict) or not image_gen.get("output_tokens"):
        return None
    model = next(
        (tool.model for tool in final.tools if tool.type == "image_generation" and tool.model),
        None,
    )
    if model is None:
        return None
    inputs = image_gen.get("input_tokens_details") or {}
    outputs = image_gen.get("output_tokens_details") or {}
    usage = Usage(
        input_tokens=int(inputs.get("text_tokens", image_gen.get("input_tokens", 0))),
        image_input_tokens=int(inputs.get("image_tokens", 0)),
        output_tokens=int(outputs.get("text_tokens", 0)),
        image_output_tokens=int(outputs.get("image_tokens", image_gen.get("output_tokens", 0))),
    )
    return model, usage


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
        # The endpoint is always given: left out, the SDK reads OPENAI_BASE_URL (D48).
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url or OWN_BASE_URLS["openai"],
            "max_retries": 0,
        }
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        super().__init__(
            model,
            _PROFILE,
            lambda: openai.AsyncOpenAI(**client_kwargs, http_client=http_client()),
        )

    def _rules(self) -> _Model:
        return _rules_of(self._model)

    def _image_rules(self) -> ImageModel | None:
        return _image_rules(self._model)

    @classmethod
    def model_facts(cls, model: str) -> AdapterFacts:
        """From the model tables: an image model only draws (``prepare`` refuses it); a chat
        model gets the profile's facts and its row's efforts (``_MODELS``)."""
        if _image_rules(model) is not None:
            return DRAWS_ONLY
        rules = _rules_of(model)
        return dataclasses.replace(
            _PROFILE.facts(),
            thinking_mode=rules.thinking_mode(),
            thinking_efforts=ordered_efforts(rules.efforts),
        )

    def _input_items(self, messages: list[dict[str, Any]]) -> list[ResponseInputItemParam]:
        """Responses input items, replaying the reasoning of this model's family only."""
        return input_items(messages, self._profile, self._profile.family(self._model))

    # ------------------------------------------------------------------
    # The contract (the I/O, the usage and the errors are the Responses core's)
    # ------------------------------------------------------------------

    def prepare(self, request: Request) -> Prepared[Params]:
        if self._image_rules() is not None:
            raise RequestError(f"{self._model} is an image model: call generate_image()")
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

    def usage(self, final: Final) -> Usage | None:
        """The turn's usage, the hosted tool's image tokens included (they count against the
        token caps; their price is the image model's, in ``provider_cost``)."""
        usage = super().usage(final)
        drawn = _hosted_image(final) if isinstance(final, SDKResponse) else None
        if usage is None or drawn is None:
            return usage
        return usage + dataclasses.replace(drawn[1], image_count=0)

    def assemble(self, final: Final, prepared: Call) -> Response:
        """The response; a turn that drew with the hosted tool gets its whole cost, the image
        priced at its image model's rates, which the turn's own model cannot price."""
        response = super().assemble(final, prepared)
        if not isinstance(final, SDKResponse) or final.usage is None:
            return response
        drawn = _hosted_image(final)
        if drawn is None:
            return response
        turn = _estimate_response_cost(self._model, _extract_usage(final.usage))
        image = _estimate_response_cost(*drawn)
        if turn is None or image is None:
            return response
        return dataclasses.replace(response, provider_cost=turn + image)

    def prepare_image(self, request: Request) -> ImagesCall:
        """An image generation on the Images API: an edit when it has input images."""
        rules = self._image_rules()
        if rules is None:
            raise RequestError(
                f"{self._model} is not an image model; generate_image takes one such as "
                "gpt-image-2.5-flare"
            )
        return images_call(self._model, request, rules)

    def image_token_bound(self, image: ImageRequest) -> int | None:
        rules = self._image_rules()
        return image_token_bound(image, rules) if rules is not None else None

    def _reasoning(self, options: Options, model: _Model) -> Reasoning:
        """The effort and the summary a request asks for.

        ``thinking_effort`` applies on its own, as on the other providers: ``"none"`` stops a
        model that reasons by default. ``thinking=True`` adds a summary of the reasoning, the
        source of the thinking blocks (https://developers.openai.com/api/docs/guides/reasoning),
        at the effort given, else ``"high"``. With neither the model reasons at its own default.
        """
        if options.thinking_budget:
            warnings.warn(
                "thinking_budget is not supported by OpenAI (it takes thinking_effort), ignoring",
                stacklevel=5,
            )
        if not options.thinking and options.thinking_effort is None:
            return {}
        if not model.efforts:
            raise RequestError(f"{self._model} does not reason: thinking is not available")
        effort = options.thinking_effort or "high"
        if effort not in model.efforts:
            raise RequestError(
                f"{self._model} takes thinking_effort in {sorted(model.efforts)}, not {effort!r}"
            )
        reasoning: Reasoning = {"effort": cast("ReasoningEffort", effort)}
        if options.thinking and effort != "none":
            reasoning["summary"] = "auto"
        return reasoning

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
        return self._answer(SDKResponse.model_construct(**body), unrequested, batch=True).response
