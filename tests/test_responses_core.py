"""Tests for _providers/_responses.py — the Responses API core, through a profile not Meta's.

Meta's side of the core is covered by ``tests/test_meta_provider.py``. These pin what a second
provider (OpenAI, O03) takes from the profile: model families for the replay guard, no
``include``, ``strict: false`` on function tools, ``tool_choice``, other hosted tools and other
error codes. The model ids are made up, so no provider's real family table is implied.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from openai.types.responses import Response as SDKResponse

from ai_arch_toolkit.core._exceptions import APIError, RateLimitError, RequestError, ResponseError
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._providers._base import Prepared, parse_options
from ai_arch_toolkit.core._providers._responses import (
    Params,
    ResponsesProfile,
    ResponsesProvider,
    _parse_sdk_response,
    input_items,
    request_params,
)
from ai_arch_toolkit.core._response import OutputSchema, Response, Usage
from tests.provider_calls import complete, prepare
from tests.wire_contract import violations

MODEL = "acme-1"
USER = {"role": "user", "content": "What is the weather in Lisbon?"}
WEATHER = {
    "name": "get_weather",
    "description": "Return the current weather for a city.",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}
PROFILE = ResponsesProfile(
    provider="Acme",
    families={"acme-1": "acme-1", "acme-2": "acme-2"},
    include=(),
    takes_tool_choice=True,
    function_strict=False,
    output_strict=True,
    hosted_tools={
        "web_search": {"type": "web_search"},
        "code_execution": {"type": "code_interpreter", "container": {"type": "auto"}},
    },
    codes={"rate_limit_exceeded": 429, "server_error": 500},
)


class AcmeProvider(ResponsesProvider):
    """The smallest adapter on the core: the request's family comes from its model."""

    def __init__(self, model: str = MODEL, client: Any = None) -> None:
        super().__init__(model, PROFILE, lambda: client or AsyncMock())

    def prepare(self, request: Request) -> Prepared[Params]:
        options = parse_options(request.kwargs, frozenset({"max_tokens"}), "Acme")
        items = input_items(request.messages, PROFILE, PROFILE.family(self._model))
        params = request_params(
            PROFILE,
            self._model,
            items,
            system=request.system,
            tools=request.tools,
            options=options,
        )
        return Prepared(params, output_schema=options.output_schema)

    def assemble(self, final: SDKResponse, prepared: Prepared[Params]) -> Response:
        return _parse_sdk_response(final, self._model, output_schema=prepared.output_schema)


def _turn(model: str) -> SDKResponse:
    """A tool turn of ``model``: its reasoning, the text that leads into the call, the call."""
    return SDKResponse.model_construct(
        id="resp_1",
        object="response",
        model=model,
        status="completed",
        output=[
            {
                "id": "rs_1",
                "type": "reasoning",
                "summary": [],
                "encrypted_content": "enc-rs_1",
            },
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "Checking.", "annotations": []}],
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": json.dumps({"city": "Lisbon"}),
                "status": "completed",
            },
        ],
    )


def _failed(code: str) -> SDKResponse:
    return SDKResponse.model_construct(
        id="resp_1",
        object="response",
        model=MODEL,
        status="failed",
        output=[],
        error={"code": code, "message": "boom"},
        usage={
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 40,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 140,
        },
    )


def _contract(params: Params) -> list[str]:
    """The Responses request contract (the net's fills for Meta's deviations only add a field a
    request leaves out)."""
    return violations("MetaProvider", params)


class TestFamilies:
    def test_a_model_takes_its_familys_name(self):
        assert PROFILE.family("acme-1-mini") == "acme-1"
        assert PROFILE.family("acme-2") == "acme-2"

    def test_a_model_without_a_family_is_its_own_without_the_snapshot(self):
        assert PROFILE.family("zeta-3-2026-01-01") == "zeta-3"
        assert PROFILE.family("zeta-3") == "zeta-3"


class TestReplay:
    @pytest.mark.parametrize(
        ("produced_by", "request_model", "replayed"),
        [
            ("acme-1-mini", "acme-1", True),
            ("acme-2", "acme-1", False),  # another family of the same provider
            ("zeta-3-2026-01-01", "zeta-3", True),  # a model no family claims, as dated
            ("zeta-4", "zeta-3", False),
            ("muse-spark-1.3", "acme-1", False),  # another provider
        ],
    )
    def test_reasoning_goes_back_only_to_the_family_that_produced_it(
        self, produced_by, request_model, replayed
    ):
        message = _parse_sdk_response(_turn(produced_by), produced_by).to_message()

        items = input_items([USER, message], PROFILE, PROFILE.family(request_model))

        kinds = [item.get("type", "user") for item in items]
        if replayed:
            assert kinds == ["user", "reasoning", "message", "function_call"]
            assert items[1]["encrypted_content"] == "enc-rs_1"
        else:
            assert kinds == ["user", "message", "function_call"]
            assert "id" not in items[1]  # rebuilt from the message's fields

    def test_a_turn_rebuilt_for_another_family_still_goes_through(self):
        message = _parse_sdk_response(_turn("acme-2"), "acme-2").to_message()

        params = prepare(AcmeProvider("acme-1"), [USER, message], tools=[WEATHER]).params

        assert [item.get("type", "user") for item in params["input"]] == [
            "user",
            "message",
            "function_call",
        ]


class TestRequest:
    def test_no_include_and_function_tools_are_strict_false(self):
        params = prepare(AcmeProvider(), [USER], tools=[WEATHER], max_tokens=64).params

        assert params == {
            "model": MODEL,
            "input": [USER],
            "store": False,
            "max_output_tokens": 64,
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Return the current weather for a city.",
                    "parameters": WEATHER["input_schema"],
                    "strict": False,
                }
            ],
        }
        assert _contract(params) == []

    @pytest.mark.parametrize(
        ("choice", "sent"),
        [
            ("auto", "auto"),
            ("none", "none"),  # the tools still go, so the model knows them
            ("required", "required"),
            ("get_weather", {"type": "function", "name": "get_weather"}),
        ],
    )
    def test_tool_choice_is_sent_as_given(self, choice, sent):
        params = prepare(AcmeProvider(), [USER], tools=[WEATHER], tool_choice=choice).params

        assert params["tool_choice"] == sent
        assert [tool["name"] for tool in params["tools"]] == ["get_weather"]
        assert _contract(params) == []

    def test_no_tool_choice_without_tools(self):
        assert "tool_choice" not in prepare(AcmeProvider(), [USER], tool_choice="required").params

    def test_hosted_tools_go_as_the_profile_names_them_and_never_share_its_dicts(self):
        code = {"_server_tool": True, "type": "code_execution"}
        params = prepare(AcmeProvider(), [USER], tools=[WEATHER, code]).params

        hosted = params["tools"][1]
        assert hosted == {"type": "code_interpreter", "container": {"type": "auto"}}
        assert _contract(params) == []
        hosted["container"]["type"] = "changed"
        assert PROFILE.hosted_tools["code_execution"] == {
            "type": "code_interpreter",
            "container": {"type": "auto"},
        }

    def test_a_server_tool_the_provider_does_not_run_is_refused(self):
        with pytest.raises(RequestError, match="'file_search': not run by Acme"):
            prepare(AcmeProvider(), [USER], tools=[{"_server_tool": True, "type": "file_search"}])


class TestFailures:
    @pytest.mark.parametrize(
        ("code", "error", "status"),
        [
            ("rate_limit_exceeded", RateLimitError, 429),
            ("server_error", APIError, 500),
            ("invalid_prompt", ResponseError, None),  # not in this profile's codes
        ],
    )
    async def test_a_failed_response_is_typed_by_the_profiles_codes(self, code, error, status):
        client = AsyncMock()
        client.responses.create = AsyncMock(return_value=_failed(code))

        with pytest.raises(error) as caught:
            await complete(AcmeProvider(client=client), [USER])

        assert getattr(caught.value, "status_code", None) == status
        if error is ResponseError:
            assert f"Acme reported a failure ({code})" in str(caught.value)
            assert caught.value.usage == Usage(input_tokens=100, output_tokens=40)


class TestOutputLimitAndFormat:
    @staticmethod
    def _params(**kwargs: Any) -> Params:
        forwarded = frozenset({"max_tokens", "max_completion_tokens"})
        options = parse_options(kwargs, forwarded, "Acme")
        return request_params(PROFILE, MODEL, [], system=None, tools=None, options=options)

    def test_max_tokens_is_the_output_limit(self):
        assert self._params(max_tokens=64)["max_output_tokens"] == 64

    def test_chat_completions_max_completion_tokens_wins(self):
        params = self._params(max_tokens=64, max_completion_tokens=128)
        assert params["max_output_tokens"] == 128
        assert not {"max_tokens", "max_completion_tokens"} & params.keys()

    def test_a_profile_that_honours_strict_sends_the_strict_subset(self):
        schema = OutputSchema(
            name="Person",
            schema={"type": "object", "properties": {"nickname": {"type": "string"}}},
        )

        text_format = self._params(output_schema=schema)["text"]["format"]

        assert text_format["strict"] is True
        assert text_format["schema"] == {
            "type": "object",
            "properties": {"nickname": {"type": "string"}},
            "required": ["nickname"],
            "additionalProperties": False,
        }
        assert schema.schema == {"type": "object", "properties": {"nickname": {"type": "string"}}}

    def test_a_non_strict_schema_goes_as_written(self):
        schema = OutputSchema(name="X", schema={"type": "object"}, strict=False)
        assert self._params(output_schema=schema)["text"]["format"]["strict"] is False


class TestLogprobs:
    def test_the_output_texts_logprobs_are_the_responses(self):
        response = SDKResponse.model_construct(
            id="resp_1",
            model=MODEL,
            status="completed",
            output=[
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "ok",
                            "annotations": [],
                            "logprobs": [
                                {
                                    "token": "ok",
                                    "bytes": [111, 107],
                                    "logprob": -0.2,
                                    "top_logprobs": [],
                                }
                            ],
                        }
                    ],
                }
            ],
        )

        parsed = _parse_sdk_response(response, MODEL)

        assert [(p.token, p.logprob) for p in parsed.logprobs] == [("ok", -0.2)]
        assert _parse_sdk_response(_turn(MODEL), MODEL).logprobs is None
