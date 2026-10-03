"""Tests for _providers/_openai.py — OpenAI's own host over the Responses API (D43).

The adapter composes the Responses core (``_responses.py``, covered with Meta's profile in
``tests/test_meta_provider.py`` and a made-up one in ``tests/test_responses_core.py``) with what is
OpenAI's: the profile, each model's rules, the Chat Completions parameters it refuses, token
counting and batches. Fixtures are built with the SDK's own lenient constructor, shaped like the
live answers of the O01 probe (2026-10-02). Compatible servers (another ``base_url`` host) keep
Chat Completions: ``tests/test_openai_compatible_provider.py``.
"""

from __future__ import annotations

import json
import warnings
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from openai.types.responses import Response as SDKResponse
from pydantic import BaseModel

from ai_arch_toolkit import LLM, RetryConfig, ToolGroup, run_tools, tool
from ai_arch_toolkit.core._exceptions import APIError, RateLimitError, RequestError, ResponseError
from ai_arch_toolkit.core._pricing import pricing
from ai_arch_toolkit.core._providers._base import on_request
from ai_arch_toolkit.core._providers._meta import MetaProvider
from ai_arch_toolkit.core._providers._openai import OpenAIProvider
from ai_arch_toolkit.core._response import (
    OutputSchema,
    ThinkingBlock,
    Usage,
    _resolve_output_schema,
)
from ai_arch_toolkit.core._server_tools import code_execution, web_search
from ai_arch_toolkit.core._tools import prepare_tools
from tests.provider_calls import complete, prepare, stream
from tests.sdk_streams import OpenAIStream

HI = [{"role": "user", "content": "Hi"}]
USER = {"role": "user", "content": "What is the weather in Lisbon?"}
LOOKUP = {"name": "lookup", "description": "d", "input_schema": {"type": "object"}}
WEATHER = {
    "name": "get_weather",
    "description": "Return the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
        "required": ["city"],
    },
}
SAMPLING = {"temperature": 0.2, "top_p": 0.9, "logprobs": True, "top_logprobs": 3}
LOGPROBS = "message.output_text.logprobs"


# ---------------------------------------------------------------------------
# Helpers — SDK objects shaped like the live API's
# ---------------------------------------------------------------------------


def _reasoning(item_id: str = "rs_1", summary: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text} for text in summary],
        "encrypted_content": f"enc-{item_id}",
    }


def _message(
    text: str,
    *,
    item_id: str = "msg_1",
    phase: str = "final_answer",
    logprobs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "phase": phase,
        "content": [
            {"type": "output_text", "text": text, "annotations": [], "logprobs": logprobs or []}
        ],
    }


def _function_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"fc_{call_id}",
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }


def _usage(input_tokens: int = 69, cached: int = 0, output_tokens: int = 20) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 8},
        "total_tokens": input_tokens + output_tokens,
    }


def _body(*output: dict[str, Any], model: str = "gpt-6-luna", status: str = "completed") -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1790000000.0,
        "model": model,
        "status": status,
        "output": list(output),
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "store": False,
        "usage": _usage(),
    }


def _sdk_response(*output: dict[str, Any], model: str = "gpt-6-luna") -> SDKResponse:
    return SDKResponse.model_construct(**_body(*output, model=model))


def _tool_turn(model: str = "gpt-6-luna") -> SDKResponse:
    """A turn that reasons, says what it will do, and calls a tool (O01's live shape)."""
    return _sdk_response(
        _reasoning(summary=("The user wants the weather.",)),
        _message("Checking Lisbon.", phase="commentary"),
        _function_call("call_1", "get_weather", {"city": "Lisbon"}),
        model=model,
    )


def _failed(code: str) -> SDKResponse:
    failed = SDKResponse.model_construct(**{**_body(status="failed"), "usage": _usage(100, 0, 40)})
    failed.error = openai.types.responses.ResponseError.construct(code=code, message="boom")
    return failed


def _client(*responses: SDKResponse) -> AsyncMock:
    client = AsyncMock()
    client.responses.create = AsyncMock(side_effect=list(responses) or [_sdk_response()])
    return client


def _provider(model: str = "gpt-6-luna", client: Any = None) -> OpenAIProvider:
    provider = OpenAIProvider(model, "test-key")
    provider._client = client or _client()
    return provider


def _params(model: str = "gpt-6-luna", messages: Any = None, **kwargs: Any) -> dict[str, Any]:
    return prepare(_provider(model), messages or HI, **kwargs).params


def _status_error(cls: type[openai.APIStatusError], status: int, **headers: str) -> Exception:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status, headers=headers, request=request)
    return cls("error", response=response, body={"message": "boom"})


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


class TestRequest:
    def test_stateless_without_include_with_flat_strict_false_tools(self):
        params = _params(
            "gpt-5.4",
            [{"role": "system", "content": "A"}, USER],
            system="Be brief.",
            tools=[WEATHER],
            max_tokens=256,
        )

        assert params == {
            "model": "gpt-5.4",
            "input": [{"role": "system", "content": "A"}, USER],
            "store": False,
            "max_output_tokens": 256,
            "instructions": "Be brief.",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Return the current weather for a city.",
                    "parameters": WEATHER["input_schema"],
                    # Left out, OpenAI rewrites the schema in strict mode and makes "unit"
                    # required (O01 check 5).
                    "strict": False,
                }
            ],
        }

    @pytest.mark.parametrize(
        ("choice", "sent"),
        [
            ("auto", "auto"),
            ("none", "none"),
            ("required", "required"),
            ("get_weather", {"type": "function", "name": "get_weather"}),
        ],
    )
    def test_tool_choice_is_sent_as_given(self, choice, sent):
        assert _params(tools=[WEATHER], tool_choice=choice)["tool_choice"] == sent

    def test_web_search_without_config_is_a_hosted_tool(self):
        # D44: the Responses API runs it; its config still belongs to C05.
        params = _params("gpt-6-astra", tools=[WEATHER, *prepare_tools([web_search()])])
        assert params["tools"][1] == {"type": "web_search"}

    @pytest.mark.parametrize(
        ("server_tool", "refusal"),
        [(web_search(max_uses=2), "config"), (code_execution(), "not run by OpenAI")],
    )
    def test_a_configured_or_another_server_tool_is_refused(self, server_tool, refusal):
        with pytest.raises(RequestError, match=refusal):
            _params(tools=prepare_tools([server_tool]))

    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-5.4-mini", "o3", "gpt-6-astra", "gpt-7"])
    def test_every_model_gets_max_output_tokens(self, model):
        params = _params(model, max_tokens=64)
        assert params["max_output_tokens"] == 64
        assert not {"max_tokens", "max_completion_tokens"} & params.keys()

    def test_chat_completions_max_completion_tokens_is_the_output_limit_and_wins(self):
        params = _params("gpt-5.4-mini", max_tokens=64, max_completion_tokens=128)
        assert params["max_output_tokens"] == 128
        assert not {"max_tokens", "max_completion_tokens"} & params.keys()

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("stop", ["\n"]),
            ("seed", 7),
            ("frequency_penalty", 0.5),
            ("presence_penalty", 0.5),
            ("response_format", {"type": "json_object"}),
        ],
    )
    async def test_chat_completions_parameters_are_refused_before_sending(self, name, value):
        # Live, stop and seed get a 400 and the penalties a 500 after ~90 s (O01 check 6); a raw
        # response_format gives way to output_schema and json_mode (D44).
        client = _client()
        with pytest.raises(RequestError, match=f"takes no {name}"):
            await complete(_provider(client=client), HI, **{name: value})
        client.responses.create.assert_not_awaited()

    def test_the_refusal_points_to_structured_output(self):
        with pytest.raises(RequestError, match="output_schema= or json_mode="):
            _params(response_format={"type": "json_object"})

    def test_a_strict_output_schema_is_sent_strict_in_the_strict_subset(self):
        class Person(BaseModel):
            name: str
            nickname: str | None = None

        params = _params("gpt-5.4", output_schema=_resolve_output_schema(Person))

        text_format = params["text"]["format"]
        assert (text_format["type"], text_format["name"], text_format["strict"]) == (
            "json_schema",
            "Person",
            True,
        )
        assert text_format["schema"]["additionalProperties"] is False
        assert text_format["schema"]["required"] == ["name", "nickname"]

    def test_a_non_strict_output_schema_is_sent_as_given(self):
        raw = {"type": "object", "properties": {"a": {"type": "string"}}}
        params = _params(output_schema=OutputSchema(name="X", schema=raw, strict=False))
        assert params["text"] == {
            "format": {"type": "json_schema", "name": "X", "schema": raw, "strict": False}
        }

    def test_json_mode(self):
        assert _params(json_mode=True)["text"] == {"format": {"type": "json_object"}}

    def test_output_schema_and_tools_coexist(self):
        schema = OutputSchema(name="X", schema={"type": "object"})
        params = _params(tools=[LOOKUP], output_schema=schema)
        assert "tools" in params
        assert "text" in params

    def test_unknown_kwargs_warn_and_known_ones_do_not(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            params = _params("gpt-4o", typo_param=True, temperature=0.5, top_p=0.9)
        assert [str(w.message) for w in caught if "typo_param" in str(w.message)]
        assert len(caught) == 1
        assert (params["temperature"], params["top_p"]) == (0.5, 0.9)


# ---------------------------------------------------------------------------
# Reasoning and each model's rules
# ---------------------------------------------------------------------------


class TestReasoning:
    def test_thinking_sends_the_effort_and_asks_for_a_summary(self):
        params = _params("gpt-5.4-mini", thinking=True, thinking_effort="medium")
        assert params["reasoning"] == {"effort": "medium", "summary": "auto"}

    def test_thinking_defaults_the_effort_to_high(self):
        assert _params("gpt-5.4-mini", thinking=True)["reasoning"]["effort"] == "high"

    def test_without_thinking_or_an_effort_no_reasoning_is_sent(self):
        assert "reasoning" not in _params("gpt-5.4-mini")

    def test_an_effort_applies_without_thinking_and_asks_for_no_summary(self):
        # As on the other providers: thinking_effort alone sets how hard the model thinks, and
        # thinking=True asks for the summary. It used to be dropped without a word.
        assert _params("gpt-5.4-mini", thinking_effort="low")["reasoning"] == {"effort": "low"}

    def test_none_alone_stops_a_model_that_reasons_by_default(self):
        params = _params("gpt-6-luna", thinking_effort="none", temperature=0.0)
        assert params["reasoning"] == {"effort": "none"}
        assert params["temperature"] == 0.0  # at "none" the sampling stays

    def test_an_effort_alone_is_checked_against_the_model(self):
        with pytest.raises(RequestError, match="thinking_effort"):
            _params("gpt-6-astra", thinking_effort="none")
        with pytest.raises(RequestError, match="does not reason"):
            _params("gpt-4.1", thinking_effort="low")

    def test_at_none_there_is_no_summary_and_the_temperature_stays(self):
        params = _params("gpt-5.4-mini", thinking=True, thinking_effort="none", temperature=0.0)
        assert params["reasoning"] == {"effort": "none"}
        assert params["temperature"] == 0.0

    @pytest.mark.parametrize(("temperature", "kept"), [(0.0, False), (1.0, True)])
    def test_reasoning_models_before_gpt_6_keep_only_the_default_temperature(
        self, temperature, kept
    ):
        params = _params(
            "gpt-5.4-mini", thinking=True, thinking_effort="medium", temperature=temperature
        )
        assert ("temperature" in params) is kept

    def test_thinking_budget_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _params("gpt-5.4-mini", thinking=True, thinking_budget=5000)
        assert len([w for w in caught if "thinking_budget" in str(w.message)]) == 1

    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-2024-08-06", "gpt-4.1-mini", "gpt-4.1"])
    def test_a_model_that_does_not_reason_refuses_thinking(self, model):
        with pytest.raises(RequestError, match="does not reason"):
            _params(model, thinking=True)

    def test_a_new_model_gets_the_current_generation_rules(self):
        params = _params("gpt-7", thinking=True, temperature=0.0)
        assert params["reasoning"] == {"effort": "high", "summary": "auto"}
        assert "temperature" not in params  # reasoning models take only the default

    @pytest.mark.parametrize(
        "model", ["gpt-5", "gpt-5.2", "gpt-5.4", "gpt-5.6-terra", "o3", "gpt-7", "gpt-6-astra"]
    )
    @pytest.mark.parametrize("effort", ["low", "high"])
    def test_tool_calls_go_at_the_effort_asked(self, model, effort):
        # Chat Completions took them only at "none" from GPT-5.4 on; the Responses API takes
        # them at any effort (https://developers.openai.com/api/docs/guides/migrate-to-responses).
        params = _params(model, tools=[LOOKUP], thinking=True, thinking_effort=effort)
        assert params["reasoning"]["effort"] == effort
        assert [tool["name"] for tool in params["tools"]] == ["lookup"]


# Each model's efforts and the effort a request without one runs at, live on the Responses API on
# 2026-10-02 (O04): one request per effort and model, and the effort the response echoes.
MEASURED_EFFORTS = {
    "gpt-5": ({"minimal", "low", "medium", "high"}, "medium"),
    "gpt-5-mini": ({"minimal", "low", "medium", "high"}, "medium"),
    "gpt-5-nano": ({"minimal", "low", "medium", "high"}, "medium"),
    "gpt-5.1": ({"none", "low", "medium", "high"}, "none"),
    "gpt-5.2": ({"none", "low", "medium", "high", "xhigh"}, "none"),
    "gpt-5.4": ({"none", "low", "medium", "high", "xhigh"}, "none"),
    "gpt-5.4-mini": ({"none", "low", "medium", "high", "xhigh"}, "none"),
    "gpt-5.4-nano": ({"none", "low", "medium", "high", "xhigh"}, "none"),
    "gpt-5.5": ({"none", "low", "medium", "high", "xhigh"}, "medium"),
    "gpt-5.6-sol": ({"none", "low", "medium", "high", "xhigh", "max"}, "medium"),
    "gpt-5.6-terra": ({"none", "low", "medium", "high", "xhigh", "max"}, "medium"),
    "gpt-5.6-luna": ({"none", "low", "medium", "high", "xhigh", "max"}, "medium"),
    "o3": ({"low", "medium", "high"}, "medium"),
    "gpt-6-astra": ({"low", "medium", "high", "xhigh", "max"}, "medium"),
    "gpt-6.1-sol": ({"low", "medium", "high", "xhigh", "max"}, "medium"),
    "gpt-6-sol": ({"none", "low", "medium", "high", "xhigh", "max"}, "medium"),
    "gpt-6-luna": ({"none", "low", "medium", "high", "xhigh", "max"}, "medium"),
    "gpt-5.6": ({"none", "low", "medium", "high", "xhigh", "max"}, "medium"),
    "gpt-5.3-codex": ({"none", "low", "medium", "high", "xhigh"}, "none"),
    "gpt-5-pro": ({"high"}, "high"),
    "gpt-5.2-pro": ({"medium", "high", "xhigh"}, "medium"),
    "gpt-5.4-pro": ({"medium", "high", "xhigh"}, "medium"),
    "gpt-5.5-pro": ({"medium", "high", "xhigh"}, "high"),
}
ALL_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class TestMeasuredEfforts:
    @pytest.mark.parametrize("model", sorted(MEASURED_EFFORTS))
    @pytest.mark.parametrize("effort", ALL_EFFORTS)
    def test_each_model_takes_the_efforts_it_took_live(self, model, effort):
        efforts, _ = MEASURED_EFFORTS[model]
        if effort in efforts:
            assert _params(model, thinking=True, thinking_effort=effort)["reasoning"][
                "effort"
            ] == (effort)
        else:
            with pytest.raises(RequestError, match="thinking_effort"):
                _params(model, thinking=True, thinking_effort=effort)

    @pytest.mark.parametrize("model", sorted(MEASURED_EFFORTS))
    def test_a_temperature_goes_only_to_a_model_that_does_not_reason_by_default(self, model):
        # Live, a model that reasons when no effort is sent refused temperature=0.0 ("Unsupported
        # parameter: 'temperature' is not supported with this model"); the LLM always sends one.
        _, default = MEASURED_EFFORTS[model]
        params = _params(model, temperature=0.0)
        assert ("temperature" in params) is (default == "none")

    def test_a_new_model_reasons_by_default_like_the_current_generation(self):
        assert "temperature" not in _params("gpt-7", temperature=0.0)


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6.1-sol"])
class TestAlwaysReasoning:
    """GPT-6 Astra and GPT-6.1 Sol take no "none" (nor "minimal") effort, so they always reason
    and take no sampling parameters (https://developers.openai.com/api/docs/models/gpt-6-astra,
    .../gpt-6.1-sol)."""

    @pytest.mark.parametrize("thinking", [False, True])
    def test_request_parameters(self, model, thinking):
        params = _params(
            model,
            max_tokens=4096,
            thinking=thinking,
            thinking_effort="xhigh",
            **{**SAMPLING, "temperature": 1.0, "top_p": 1.0},
        )
        assert params["max_output_tokens"] == 4096
        assert not {"temperature", "top_p", "top_logprobs", "include"} & params.keys()
        assert params["reasoning"] == (
            {"effort": "xhigh", "summary": "auto"} if thinking else {"effort": "xhigh"}
        )

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
    def test_every_documented_effort_is_sent(self, model, effort):
        # "max" is on the model pages; Chat Completions refused it, the Responses API takes it
        # (live on gpt-6.1-sol, O01 check 8).
        assert _params(model, thinking=True, thinking_effort=effort)["reasoning"]["effort"] == (
            effort
        )

    @pytest.mark.parametrize("effort", ["none", "minimal", "ultra"])
    def test_invalid_reasoning_effort(self, model, effort):
        with pytest.raises(RequestError, match="thinking_effort"):
            _params(model, thinking=True, thinking_effort=effort)

    @pytest.mark.parametrize("thinking", [False, True])
    def test_they_call_tools(self, model, thinking):
        # They took no tools through Chat Completions; through the Responses API they do.
        params = _params(model, tools=[LOOKUP], thinking=thinking, thinking_effort="low")
        assert [tool["name"] for tool in params["tools"]] == ["lookup"]
        assert params["reasoning"]["effort"] == "low"


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
class TestSolAndLuna:
    """GPT-6 Sol and Luna reason at "medium" unless sent "none", and take sampling only at "none"
    (https://developers.openai.com/api/docs/models/gpt-6-sol,
    https://developers.openai.com/api/docs/guides/latest-model)."""

    def test_by_default_they_reason_so_sampling_is_dropped(self, model):
        params = _params(model, max_tokens=64, **SAMPLING)
        assert "reasoning" not in params
        assert not {"temperature", "top_p", "top_logprobs", "include"} & params.keys()
        assert params["max_output_tokens"] == 64

    @pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh", "max"])
    def test_every_documented_effort_is_sent(self, model, effort):
        assert _params(model, thinking=True, thinking_effort=effort)["reasoning"]["effort"] == (
            effort
        )

    def test_minimal_is_not_among_their_efforts(self, model):
        with pytest.raises(RequestError, match="thinking_effort"):
            _params(model, thinking=True, thinking_effort="minimal")

    def test_at_none_they_sample_and_return_logprobs(self, model):
        params = _params(model, thinking=True, thinking_effort="none", **SAMPLING)
        assert params["reasoning"] == {"effort": "none"}
        assert (params["temperature"], params["top_p"], params["top_logprobs"]) == (0.2, 0.9, 3)
        assert params["include"] == [LOGPROBS]

    def test_tool_calls_without_thinking_run_at_the_models_default(self, model):
        # Chat Completions forced them to "none"; a request with tools now runs like one without.
        params = _params(model, tools=[LOOKUP], tool_choice="auto", temperature=0.0)
        assert "reasoning" not in params
        assert params["tools"][0]["name"] == "lookup"
        assert params["tool_choice"] == "auto"
        assert "temperature" not in params  # medium by default: they reason

    @pytest.mark.parametrize("effort", ["none", "low", "xhigh"])
    def test_tool_calls_go_at_any_effort(self, model, effort):
        params = _params(model, tools=[LOOKUP], thinking=True, thinking_effort=effort)
        assert params["reasoning"]["effort"] == effort
        assert "tools" in params


class TestLogprobs:
    def test_a_model_that_does_not_reason_returns_them(self):
        params = _params("gpt-4o", logprobs=True, top_logprobs=2)
        assert (params["include"], params["top_logprobs"]) == ([LOGPROBS], 2)

    def test_they_are_assembled_from_the_output_text(self):
        tokens = [
            {"token": "Hi", "bytes": [72, 105], "logprob": -0.1, "top_logprobs": []},
            {"token": "!", "bytes": [33], "logprob": -0.4, "top_logprobs": []},
        ]
        response = _provider("gpt-4o").assemble(
            _sdk_response(_message("Hi!", logprobs=tokens), model="gpt-4o"),
            prepare(_provider("gpt-4o"), HI, logprobs=True),
        )
        assert [(p.token, p.logprob) for p in response.logprobs] == [("Hi", -0.1), ("!", -0.4)]

    def test_none_when_not_asked(self):
        response = _provider().assemble(_sdk_response(_message("Hi")), prepare(_provider(), HI))
        assert response.logprobs is None


# ---------------------------------------------------------------------------
# Answers, replay and failures
# ---------------------------------------------------------------------------


class TestComplete:
    async def test_the_answer_with_summaries_as_thinking(self):
        client = _client(_tool_turn())
        response = await complete(_provider(client=client), [USER], tools=[WEATHER], thinking=True)

        assert response.text == "Checking Lisbon."
        assert response.thinking == (ThinkingBlock(text="The user wants the weather."),)
        assert [(c.id, c.name, c.input) for c in response.tool_calls] == [
            ("call_1", "get_weather", {"city": "Lisbon"})
        ]
        assert response.usage == Usage(input_tokens=69, output_tokens=20)
        assert response.cost is not None and response.cost > 0
        assert response.stop_reason == "completed"

    async def test_a_tool_loop_through_llm_replays_the_encrypted_reasoning(self):
        @tool
        def get_weather(city: str) -> str:
            """Return the weather.

            Args:
                city: City name.
            """
            return "Sunny, 24C"

        client = _client(_tool_turn(), _sdk_response(_message("Sunny, 24C in Lisbon.")))
        group = ToolGroup(get_weather)
        async with LLM("gpt-6-luna", api_key="test-key") as llm:
            llm._provider._client = client  # type: ignore[attr-defined]
            first = await llm.complete([USER], tools=group, thinking=True, thinking_effort="low")
            results = await run_tools(first, group)
            history = [USER, first.to_message(), *results]
            final = await llm.complete(history, tools=group, thinking=True, thinking_effort="low")

        sent = client.responses.create.call_args_list[1].kwargs
        assert [item.get("type", "user") for item in sent["input"]] == [
            "user",
            "reasoning",
            "message",
            "function_call",
            "function_call_output",
        ]
        assert sent["input"][1]["encrypted_content"] == "enc-rs_1"
        assert sent["input"][2]["phase"] == "commentary"
        assert (sent["store"], sent["reasoning"]) == (False, {"effort": "low", "summary": "auto"})
        assert final.text == "Sunny, 24C in Lisbon."

    @pytest.mark.parametrize(
        ("produced_by", "requested_by", "replayed"),
        [
            ("gpt-6-luna", "gpt-6-luna", True),
            ("gpt-6-luna", "gpt-6-astra", True),  # one GPT-6 family: Astra, Sol, Luna
            ("gpt-5.6-terra", "gpt-5.6-sol", True),  # the reasoning guide's own example
            ("gpt-5.6-terra", "gpt-5.5", False),  # "does not carry between" 5.6 and 5.5
            ("gpt-5.4-2026-03-05", "gpt-5.4-mini", True),
            ("gpt-6-luna", "gpt-6.1-sol", False),
            ("o3", "o3-2025-04-16", True),
            ("o3", "gpt-5", False),
        ],
    )
    def test_reasoning_goes_back_only_within_its_family(self, produced_by, requested_by, replayed):
        turn = _provider(produced_by).assemble(_tool_turn(produced_by), prepare(_provider(), HI))

        items = _params(requested_by, [USER, turn.to_message()])["input"]

        assert ("reasoning" in [item.get("type") for item in items]) is replayed

    def test_a_meta_turn_is_rebuilt_without_its_reasoning(self):
        meta = MetaProvider("muse-spark-1.3", "test-key")
        turn = meta.assemble(_tool_turn("muse-spark-1.3"), prepare(meta, HI))

        items = _params("gpt-6-luna", [USER, turn.to_message()])["input"]

        assert [item.get("type", "user") for item in items] == ["user", "message", "function_call"]
        assert "id" not in items[1]

    @pytest.mark.parametrize(
        ("code", "error", "status"),
        [
            ("rate_limit_exceeded", RateLimitError, 429),
            ("server_error", APIError, 500),
            ("invalid_prompt", ResponseError, None),  # documented with no status
        ],
    )
    async def test_a_failed_response_is_typed_by_openais_codes(self, code, error, status):
        with pytest.raises(error) as caught:
            await complete(_provider(client=_client(_failed(code))), HI)
        assert getattr(caught.value, "status_code", None) == status
        if error is APIError:
            assert caught.value.usage == Usage(input_tokens=100, output_tokens=40)

    async def test_rate_limit_error(self):
        client = AsyncMock()
        client.responses.create.side_effect = _status_error(
            openai.RateLimitError, 429, **{"retry-after": "3.0"}
        )
        with pytest.raises(RateLimitError) as caught:
            await complete(_provider(client=client), HI)
        assert (caught.value.status_code, caught.value.retry_after) == (429, 3.0)

    async def test_status_error(self):
        client = AsyncMock()
        client.responses.create.side_effect = _status_error(openai.InternalServerError, 500)
        with pytest.raises(APIError) as caught:
            await complete(_provider(client=client), HI)
        assert caught.value.status_code == 500

    @pytest.mark.parametrize(
        ("sdk_error", "expected"),
        [(openai.APIConnectionError, ConnectionError), (openai.APITimeoutError, TimeoutError)],
    )
    async def test_network_failures_become_builtin_errors(self, sdk_error, expected):
        error = sdk_error(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
        client = AsyncMock()
        client.responses.create.side_effect = error
        provider = _provider(client=client)
        with pytest.raises(expected):
            await complete(provider, HI)
        with pytest.raises(expected):
            await stream(provider, HI)

    async def test_a_dropped_connection_is_retried_by_llm(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        client = AsyncMock()
        client.responses.create.side_effect = [
            openai.APIConnectionError(request=request),
            _sdk_response(_message("recovered"), model="gpt-4o"),
        ]
        async with LLM("gpt-4o", api_key="test-key", retry=RetryConfig(base_delay=0.01)) as llm:
            llm._provider._client = client  # type: ignore[attr-defined]
            response = await llm.complete("Hi")

        assert response.text == "recovered"
        assert client.responses.create.await_count == 2


class TestStream:
    async def test_summaries_stream_as_thinking_and_the_final_turn_is_assembled(self):
        final = _tool_turn()
        events = [
            {"type": "response.created"},
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_1",
                "summary_index": 0,
                "delta": "The user wants",
            },
            {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Checking "},
            {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Lisbon."},
            {"type": "response.completed", "response": final},
        ]
        client = AsyncMock()
        client.responses.create = AsyncMock(
            return_value=OpenAIStream([SimpleNamespace(**event) for event in events])
        )

        streamed, response = await stream(
            _provider(client=client), [USER], tools=[WEATHER], thinking=True
        )

        thinking = [e.thinking.text for e in streamed if e.kind == "thinking"]
        assert thinking == ["The user wants"]
        assert "".join(e.text for e in streamed if e.kind == "text") == "Checking Lisbon."
        assert [e.tool_call.id for e in streamed if e.kind == "tool_call"] == ["call_1"]
        assert response.raw is final
        assert client.responses.create.call_args.kwargs["stream"] is True

    async def test_an_error_event_is_typed_by_openais_codes(self):
        events = [{"type": "error", "code": "server_error", "message": "busy"}]
        client = AsyncMock()
        client.responses.create = AsyncMock(
            return_value=OpenAIStream([SimpleNamespace(**event) for event in events])
        )
        with pytest.raises(APIError) as caught:
            await stream(_provider(client=client), HI)
        assert caught.value.status_code == 500


class TestCountTokens:
    async def test_counts_with_the_input_tokens_endpoint(self):
        client = AsyncMock()
        client.responses.input_tokens.count = AsyncMock(
            return_value=SimpleNamespace(input_tokens=69)
        )

        count = await _provider(client=client).count_tokens(
            [USER], system="Be brief.", tools=[WEATHER, *prepare_tools([web_search()])]
        )

        assert count == 69
        sent = client.responses.input_tokens.count.call_args.kwargs
        assert (sent["model"], sent["instructions"]) == ("gpt-6-luna", "Be brief.")
        assert sent["input"] == [USER]
        assert [(t["name"], t["strict"]) for t in sent["tools"]] == [("get_weather", False)]


# ---------------------------------------------------------------------------
# Batch: /v1/responses, and the batches submitted to Chat Completions before
# ---------------------------------------------------------------------------


def _batch_client(endpoint: str, *lines: dict[str, Any]) -> MagicMock:
    client = MagicMock()
    client.files.create = AsyncMock(return_value=MagicMock(id="file-abc"))
    client.files.content = AsyncMock(
        return_value=MagicMock(text="\n".join(json.dumps(line) for line in lines))
    )
    client.batches.create = AsyncMock(return_value=MagicMock(id="batch-1"))
    client.batches.retrieve = AsyncMock(
        return_value=MagicMock(endpoint=endpoint, output_file_id="file-out", status="in_progress")
    )
    return client


class TestBatch:
    async def test_submits_responses_requests_on_the_responses_endpoint(self):
        client = _batch_client("/v1/responses")
        provider = _provider("gpt-6-luna", client)

        batch_id = await provider.batch_submit(
            [
                {"custom_id": "req-1", "messages": HI, "system": "Be brief."},
                {"custom_id": "req-2", "messages": HI, "kwargs": {"thinking": True}},
            ]
        )

        assert batch_id == "batch-1"
        assert client.batches.create.call_args.kwargs == {
            "input_file_id": "file-abc",
            "endpoint": "/v1/responses",
            "completion_window": "24h",
        }
        file = client.files.create.call_args.kwargs["file"]
        lines = [json.loads(line) for line in file.read().decode().splitlines()]
        assert [(line["custom_id"], line["url"]) for line in lines] == [
            ("req-1", "/v1/responses"),
            ("req-2", "/v1/responses"),
        ]
        assert lines[0]["body"] == {
            "model": "gpt-6-luna",
            "input": HI,
            "store": False,
            "max_output_tokens": 4096,
            "instructions": "Be brief.",
        }
        assert lines[1]["body"]["reasoning"] == {"effort": "high", "summary": "auto"}
        assert await provider.batch_status("batch-1") == "in_progress"

    async def test_responses_results_are_assembled_as_a_calls_answer(self):
        body = _body(_reasoning(), _message("Hello!"))
        provider = _provider(
            "gpt-6-luna",
            _batch_client("/v1/responses", {"custom_id": "req-1", "response": {"body": body}}),
        )

        [result] = await provider.batch_results("batch-1")

        assert result.custom_id == "req-1"
        assert result.response is not None
        assert result.response.text == "Hello!"
        assert result.response.usage == Usage(input_tokens=69, output_tokens=20)
        # Priced at the batch rates, half the standard ones.
        assert result.response.cost == pytest.approx(
            pricing.estimate_cost("gpt-6-luna", 69, 20, is_batch=True)
        )
        # The SDK's response: the turn replays its reasoning like a call's.
        assert isinstance(result.response.raw, SDKResponse)

    async def test_a_batch_submitted_to_chat_completions_still_reads(self):
        body = {
            "id": "chatcmpl-1",
            "model": "gpt-4o",
            "choices": [{"message": {"content": "Hi there"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        error = {"code": "rate_limit_exceeded", "message": "slow"}
        provider = _provider(
            "gpt-4o",
            _batch_client(
                "/v1/chat/completions",
                {"custom_id": "req-1", "response": {"body": body}},
                {"custom_id": "req-2", "error": error},
            ),
        )

        ok, failed = await provider.batch_results("batch-0")

        assert ok.response is not None
        assert (ok.response.text, ok.response.stop_reason) == ("Hi there", "stop")
        assert ok.response.usage == Usage(input_tokens=10, output_tokens=5)
        assert ok.response.cost == pytest.approx(
            pricing.estimate_cost("gpt-4o", 10, 5, is_batch=True)
        )
        assert failed.error is not None and "rate_limit_exceeded" in failed.error


class TestClient:
    @patch("ai_arch_toolkit.core._providers._openai.openai.AsyncOpenAI")
    def test_disables_hidden_sdk_retries(self, client_cls):
        OpenAIProvider("gpt-4o", "test-key")

        kwargs = client_cls.call_args.kwargs
        assert (kwargs["api_key"], kwargs["max_retries"]) == ("test-key", 0)
        # Given explicitly: left out, the SDK would read OPENAI_BASE_URL (D48).
        assert kwargs["base_url"] == "https://api.openai.com/v1"
        # The HTTP client marks the moment a request is handed to the transport.
        assert kwargs["http_client"].event_hooks["request"] == [on_request]

    def test_base_url_and_timeout(self):
        client = OpenAIProvider("gpt-4o", "k", base_url="https://api.openai.com/v1", timeout=5.0)
        assert str(client._client.base_url).rstrip("/") == "https://api.openai.com/v1"
        assert client._client.timeout == 5.0

    async def test_close_and_context_manager(self):
        async with OpenAIProvider("gpt-4o", "test-key") as provider:
            assert provider._client is not None
        await OpenAIProvider("gpt-4o", "test-key").close()
