"""Tests for _providers/_meta.py — Meta Model API adapter over the openai Responses API.

Fixtures follow payloads recorded from the live API (muse-spark-1.3, 2026-09-13), built with
the SDK's own lenient constructor so the adapter sees the same objects it gets in production.
"""

from __future__ import annotations

import json
import sys
import warnings
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import httpx2
import openai
import pytest
from openai.types.responses import Response as SDKResponse
from pydantic import BaseModel

from ai_arch_toolkit import LLM, RetryConfig, ToolGroup, run_tools, tool
from ai_arch_toolkit.core._content import CachePart, DocumentPart, ImagePart
from ai_arch_toolkit.core._exceptions import (
    APIError,
    ProviderTimeout,
    RateLimitError,
    RequestError,
    ResponseError,
    TransportError,
)
from ai_arch_toolkit.core._providers._base import on_request
from ai_arch_toolkit.core._providers._meta import (
    DEFAULT_BASE_URL,
    MetaProvider,
    _input_items,
    _parse_sdk_response,
)
from ai_arch_toolkit.core._response import OutputSchema, ThinkingBlock, ToolCall, Usage
from ai_arch_toolkit.core._server_tools import code_execution, web_search
from tests.provider_calls import assembled, complete, prepare, stream
from tests.sdk_streams import OpenAIStream

MODEL = "muse-spark-1.3"
USER = {"role": "user", "content": "What is the weather in Lisbon?"}
WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Return the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


# ---------------------------------------------------------------------------
# Helpers — SDK objects shaped like the live API's
# ---------------------------------------------------------------------------


def _reasoning(item_id: str = "rs_1", summary: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text} for text in summary],
        "content": None,
        "encrypted_content": f"enc-{item_id}",
        "status": "completed",
    }


def _message(
    text: str,
    *,
    item_id: str = "msg_1",
    phase: str | None = None,
    annotations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "phase": phase,
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": annotations or [],
                "logprobs": [],
            }
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


def _sdk_response(
    *output: dict[str, Any],
    status: str = "completed",
    incomplete: str | None = None,
    input_tokens: int = 553,
    cached: int = 113,
    output_tokens: int = 113,
    reasoning_tokens: int = 43,
) -> SDKResponse:
    return SDKResponse.model_construct(
        **{
            "id": "resp_1",
            "object": "response",
            "created_at": 1789270716.0,
            "model": MODEL,
            "status": status,
            "incomplete_details": {"reason": incomplete} if incomplete else None,
            "output": list(output),
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "store": False,
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": {"cache_write_tokens": None, "cached_tokens": cached},
                "output_tokens": output_tokens,
                "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
                "total_tokens": input_tokens + output_tokens,
            },
        }
    )


def _tool_turn() -> SDKResponse:
    """A turn that says what it will do, then calls a tool (the live API's usual shape)."""
    return _sdk_response(
        _reasoning(),
        _message("I'll check the weather in Lisbon.", phase="commentary"),
        _function_call("call_1", "get_weather", {"city": "Lisbon"}),
    )


def _provider(client: Any = None) -> MetaProvider:
    provider = MetaProvider(MODEL, "test-key")
    provider._client = client or AsyncMock()
    return provider


def _request(**kwargs: Any) -> dict[str, Any]:
    messages = kwargs.pop("messages", [USER])
    return prepare(_provider(), messages, **kwargs).params


def _failed(code: str | None, message: str = "boom") -> SDKResponse:
    """A response whose status is ``failed``, with the usage it reported."""
    failed = _sdk_response(status="failed", input_tokens=100, cached=0, output_tokens=40)
    failed.error = openai.types.responses.ResponseError.construct(code=code, message=message)
    return failed


def _status_error(cls: type[openai.APIStatusError], status: int, **headers: str) -> Exception:
    request = httpx.Request("POST", f"{DEFAULT_BASE_URL}/responses")
    response = httpx.Response(status, headers=headers, request=request)
    return cls("error", response=response, body={"message": "boom"})


def _events(*events: dict[str, Any]) -> AsyncMock:
    """A client whose streamed ``responses.create`` yields the given events."""

    client = AsyncMock()
    stream = OpenAIStream([SimpleNamespace(**event) for event in events])
    client.responses.create = AsyncMock(return_value=stream)
    return client


def _completed(response: SDKResponse, kind: str = "response.completed") -> dict[str, Any]:
    return {"type": kind, "response": response}


# ---------------------------------------------------------------------------
# Input conversion
# ---------------------------------------------------------------------------


class TestInputItems:
    def test_system_messages_keep_their_positions(self):
        items = _input_items(
            [
                {"role": "system", "content": "A"},
                USER,
                {"role": "system", "content": ["B", CachePart(content="C")]},
            ]
        )
        assert items == [
            {"role": "system", "content": "A"},
            USER,
            {"role": "system", "content": "B\n\nC"},
        ]

    def test_multimodal_user_content(self):
        items = _input_items(
            [
                {
                    "role": "user",
                    "content": [
                        "Compare these.",
                        ImagePart(source=b"\x89PNG", media_type="image/png"),
                        ImagePart(source="https://example.com/cat.jpg"),
                        DocumentPart(source=b"%PDF-1.4", name="report.pdf"),
                        DocumentPart(source="https://example.com/paper.pdf"),
                        CachePart(content="Long shared context."),
                    ],
                }
            ]
        )
        assert items == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Compare these."},
                    {"type": "input_image", "image_url": "data:image/png;base64,iVBORw=="},
                    {"type": "input_image", "image_url": "https://example.com/cat.jpg"},
                    {
                        "type": "input_file",
                        "filename": "report.pdf",
                        "file_data": "data:application/pdf;base64,JVBERi0xLjQ=",
                    },
                    {"type": "input_file", "file_url": "https://example.com/paper.pdf"},
                    {"type": "input_text", "text": "Long shared context."},
                ],
            }
        ]

    def test_tool_result_becomes_function_call_output(self):
        items = _input_items(
            [{"role": "tool", "tool_use_id": "call_1", "name": "get_weather", "content": 24}]
        )
        assert items == [{"type": "function_call_output", "call_id": "call_1", "output": "24"}]

    def test_assistant_tool_turn_without_raw_is_rebuilt_as_commentary(self):
        items = _input_items(
            [
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [{"id": "c1", "name": "get_weather", "input": {"city": "X"}}],
                }
            ]
        )
        assert items == [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Let me check."}],
                "phase": "commentary",
            },
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "get_weather",
                "arguments": '{"city": "X"}',
            },
        ]

    def test_assistant_answer_without_raw_has_no_phase(self):
        items = _input_items([{"role": "assistant", "content": "Sunny."}])
        assert items == [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Sunny."}],
            }
        ]

    def test_to_message_replays_the_encrypted_reasoning(self):
        message = _parse_sdk_response(_tool_turn(), MODEL).to_message()

        items = _input_items([USER, message])

        assert [item.get("type") for item in items[1:]] == [
            "reasoning",
            "message",
            "function_call",
        ]
        assert items[1]["encrypted_content"] == "enc-rs_1"
        assert items[2]["phase"] == "commentary"
        assert items[3]["call_id"] == "call_1"

    def test_edited_text_drops_the_stale_reasoning(self):
        message = _parse_sdk_response(_tool_turn(), MODEL).to_message()
        message["content"] = "Something else."

        items = _input_items([message])

        assert [item["type"] for item in items] == ["message", "function_call"]
        assert items[0]["content"][0]["text"] == "Something else."

    def test_edited_tool_arguments_drop_the_stale_reasoning(self):
        message = _parse_sdk_response(_tool_turn(), MODEL).to_message()
        message["tool_calls"][0]["input"] = {"city": "Porto"}

        items = _input_items([message])

        assert [item["type"] for item in items] == ["message", "function_call"]
        assert items[1]["arguments"] == '{"city": "Porto"}'

    def test_raw_from_another_provider_is_ignored(self):
        chat_completion = SimpleNamespace(object="chat.completion", choices=[])
        message = {"role": "assistant", "content": "Hi.", "_raw": chat_completion}

        assert _input_items([message]) == [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi."}],
            }
        ]

    def test_trailing_reasoning_of_an_incomplete_turn_is_not_replayed(self):
        truncated = _sdk_response(
            _message("Partial"),
            _reasoning("rs_2"),
            status="incomplete",
            incomplete="max_output_tokens",
        )
        message = _parse_sdk_response(truncated, MODEL).to_message()

        assert [item["type"] for item in _input_items([message])] == ["message"]

    def test_reasoning_without_encrypted_content_is_not_replayed(self):
        # Meta answers 400 "reasoning item was not found" when store=false and the content is
        # missing, e.g. behind a gateway that drops the include parameter.
        bare = {**_reasoning(), "encrypted_content": None}
        turn = _sdk_response(
            bare,
            _message("Checking.", phase="commentary"),
            _function_call("call_1", "get_weather", {"city": "Lisbon"}),
        )
        message = _parse_sdk_response(turn, MODEL).to_message()

        items = _input_items([message])

        assert [item["type"] for item in items] == ["message", "function_call"]


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


class TestBuildRequest:
    def test_stateless_request_with_encrypted_reasoning(self):
        request = _request(system="Be brief.", max_tokens=512, temperature=1.0, top_p=0.9)
        assert request == {
            "model": MODEL,
            "input": [USER],
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "instructions": "Be brief.",
            "max_output_tokens": 512,
            "temperature": 1.0,
            "top_p": 0.9,
        }

    def test_thinking_asks_for_a_summary_and_effort_applies_on_its_own(self):
        assert _request(thinking=True)["reasoning"] == {"summary": "auto"}
        assert _request(thinking_effort="low")["reasoning"] == {"effort": "low"}
        assert _request(thinking=True, thinking_effort="max")["reasoning"] == {
            "effort": "max",
            "summary": "auto",
        }
        assert "reasoning" not in _request()

    def test_thinking_budget_warns(self):
        with pytest.warns(UserWarning, match="thinking_budget is not supported by Meta"):
            request = _request(thinking=True, thinking_budget=2048)
        assert request["reasoning"] == {"summary": "auto"}

    def test_unknown_parameters_warn_and_are_not_sent(self):
        with pytest.warns(
            UserWarning, match=r"Unknown parameter\(s\) ignored for Meta: \['stop'\]"
        ):
            request = _request(stop=["\n"])
        assert "stop" not in request

    def test_logprobs_are_refused(self):
        # "logprobs: true returns HTTP 400" (https://dev.meta.ai/docs/reasoning).
        with pytest.raises(RequestError, match="logprobs"):
            _request(logprobs=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert "logprobs" not in _request(logprobs=False)

    @pytest.mark.parametrize(
        ("model", "effort", "sent"),
        [
            ("muse-spark-1.3", "max", True),
            ("muse-spark-1.4", "max", True),  # a newer model gets every effort
            ("muse-spark-1.3-contributor", "max", False),
            ("muse-spark-1.2", "max", False),
            ("muse-spark-1.1", "none", True),
            ("muse-spark-1.3", "turbo", False),
        ],
    )
    def test_the_effort_must_be_one_the_model_takes(self, model, effort, sent):
        # "max" is for standard-tier muse-spark-1.3 only (https://dev.meta.ai/docs/reasoning).
        provider = MetaProvider(model, "test-key")
        if sent:
            reasoning = prepare(provider, [USER], thinking_effort=effort).params["reasoning"]
            assert reasoning == {"effort": effort}
        else:
            with pytest.raises(RequestError, match=effort):
                prepare(provider, [USER], thinking_effort=effort)

    def test_a_role_the_responses_api_does_not_have_is_refused(self):
        with pytest.raises(RequestError, match="tool"):
            _request(messages=[{"role": "tool", "content": "orphan"}])

    def test_function_tools_are_flat_and_web_search_is_hosted(self):
        wire_web_search = {"_server_tool": True, "type": web_search().type}
        request = _request(tools=[WEATHER_TOOL, wire_web_search])
        assert request["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Return the current weather for a city.",
                "parameters": WEATHER_TOOL["input_schema"],
            },
            {"type": "web_search"},
        ]

    @pytest.mark.parametrize(
        "server_tool",
        [
            {"_server_tool": True, "type": code_execution().type},
            {"_server_tool": True, "type": web_search().type, "max_uses": 3},
        ],
        ids=["code execution", "a config"],
    )
    def test_a_server_tool_meta_does_not_run_is_refused(self, server_tool):
        # Both used to be dropped with a warning (a server tool's config belongs to C05).
        with pytest.raises(RequestError, match=server_tool["type"]):
            _request(tools=[WEATHER_TOOL, server_tool])

    def test_tool_choice_auto_is_the_default_and_none_sends_no_tools(self):
        auto = _request(tools=[WEATHER_TOOL], tool_choice="auto")
        assert auto["tools"] and "tool_choice" not in auto
        assert "tools" not in _request(tools=[WEATHER_TOOL], tool_choice="none")

    @pytest.mark.parametrize("choice", ["required", "get_weather"])
    def test_forced_tool_choice_is_rejected_before_the_call(self, choice):
        with pytest.raises(ValueError, match="only supports tool_choice='auto'"):
            _request(tools=[WEATHER_TOOL], tool_choice=choice)

    def test_output_schema_is_sent_non_strict(self):
        schema = OutputSchema(name="Legs", schema={"type": "object"}, strict=True)
        assert _request(output_schema=schema)["text"] == {
            "format": {
                "type": "json_schema",
                "name": "Legs",
                "schema": {"type": "object"},
                "strict": False,
            }
        }

    def test_json_mode(self):
        assert _request(json_mode=True)["text"] == {"format": {"type": "json_object"}}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class Legs(BaseModel):
    animal: str
    legs: int


class TestParseResponse:
    def test_tool_turn(self):
        response = assembled(_provider(), _tool_turn())

        assert response.text == "I'll check the weather in Lisbon."
        assert response.tool_calls == (
            ToolCall(id="call_1", name="get_weather", input={"city": "Lisbon"}),
        )
        # Cached input is split out; reasoning tokens are already inside output_tokens.
        assert response.usage == Usage(input_tokens=440, output_tokens=113, cache_read_tokens=113)
        assert response.cost == pytest.approx((440 * 1.25 + 113 * 4.25 + 113 * 0.15) / 1e6)
        assert response.stop_reason == "completed"
        assert response.response_id == "resp_1"
        assert response.model == MODEL

    def test_messages_join_with_a_blank_line_and_summaries_become_thinking(self):
        response = _parse_sdk_response(
            _sdk_response(
                _reasoning("rs_1", ("Planning the search.",)),
                _message("I'll search.", item_id="m1", phase="commentary"),
                _reasoning("rs_2", ("Checking a source.", "Confirming.")),
                _message("Dembélé won.", item_id="m2"),
            ),
            MODEL,
        )
        assert response.text == "I'll search.\n\nDembélé won."
        assert response.thinking == (
            ThinkingBlock(text="Planning the search."),
            ThinkingBlock(text="Checking a source.\n\nConfirming."),
        )

    def test_incomplete_response_reports_why(self):
        response = _parse_sdk_response(
            _sdk_response(_reasoning(), status="incomplete", incomplete="max_output_tokens"),
            MODEL,
        )
        assert response.stop_reason == "max_output_tokens"
        assert response.text == ""

    def test_url_citations(self):
        annotation = {
            "type": "url_citation",
            "start_index": 0,
            "end_index": 12,
            "url": "https://example.com/ballon",
            "title": "Ballon d'Or",
        }
        response = _parse_sdk_response(
            _sdk_response(_message("Dembélé won the award.", annotations=[annotation])), MODEL
        )
        (citation,) = response.citations
        assert (citation.text, citation.url, citation.title) == (
            "Dembélé won ",
            "https://example.com/ballon",
            "Ballon d'Or",
        )

    def test_structured_output_parses_the_final_message(self):
        schema = OutputSchema(name="Legs", schema={"type": "object"}, model_class=Legs)
        response = _parse_sdk_response(
            _sdk_response(
                _message("Let me think.", item_id="m1", phase="commentary"),
                _message('{"animal": "spider", "legs": 8}', item_id="m2"),
            ),
            MODEL,
            output_schema=schema,
        )
        assert response.parsed == Legs(animal="spider", legs=8)

    def test_unparseable_structured_output_is_none(self):
        schema = OutputSchema(name="Legs", schema={"type": "object"}, model_class=Legs)
        response = _parse_sdk_response(
            _sdk_response(_message("eight")), MODEL, output_schema=schema
        )
        assert response.parsed is None

    def test_refusal(self):
        refusal = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "refusal", "refusal": "I can't help with that."}],
        }
        response = _parse_sdk_response(_sdk_response(refusal), MODEL)
        assert response.text == "I can't help with that."
        assert response.stop_reason == "refusal"


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------


class TestComplete:
    async def test_sends_the_request_and_parses_the_response(self):
        client = AsyncMock()
        client.responses.create = AsyncMock(return_value=_tool_turn())
        provider = _provider(client)

        response = await complete(provider, [USER], tools=[WEATHER_TOOL], max_tokens=256)

        sent = client.responses.create.call_args.kwargs
        assert sent["max_output_tokens"] == 256
        assert sent["store"] is False
        assert response.tool_calls[0].name == "get_weather"

    async def test_rate_limit_error(self):
        client = AsyncMock()
        client.responses.create.side_effect = _status_error(
            openai.RateLimitError, 429, **{"retry-after": "15"}
        )
        with pytest.raises(RateLimitError) as caught:
            await complete(_provider(client), [USER])
        assert caught.value.status_code == 429
        assert caught.value.retry_after == 15.0

    async def test_status_error(self):
        client = AsyncMock()
        client.responses.create.side_effect = _status_error(openai.InternalServerError, 503)
        with pytest.raises(APIError) as caught:
            await complete(_provider(client), [USER])
        assert caught.value.status_code == 503

    async def test_a_failed_response_raises_with_the_usage_it_reported(self):
        client = AsyncMock()
        client.responses.create = AsyncMock(return_value=_failed("server_error"))
        with pytest.raises(APIError) as caught:
            await complete(_provider(client), [USER])
        assert caught.value.status_code == 500
        assert caught.value.usage == Usage(input_tokens=100, output_tokens=40)

    async def test_a_tool_loop_through_llm_replays_the_reasoning(self):
        @tool
        def get_weather(city: str) -> str:
            """Return the weather.

            Args:
                city: City name.
            """
            return "Sunny, 24C"

        client = AsyncMock()
        client.responses.create = AsyncMock(
            side_effect=[_tool_turn(), _sdk_response(_message("Sunny, 24C in Lisbon."))]
        )
        group = ToolGroup(get_weather)
        async with LLM(MODEL, api_key="test-key") as llm:
            llm._provider._client = client  # type: ignore[attr-defined]
            first = await llm.complete([USER], tools=group)
            results = await run_tools(first, group)
            final = await llm.complete([USER, first.to_message(), *results], tools=group)

        replayed = client.responses.create.call_args_list[1].kwargs["input"]
        assert [item.get("type", "user") for item in replayed] == [
            "user",
            "reasoning",
            "message",
            "function_call",
            "function_call_output",
        ]
        assert replayed[1]["encrypted_content"] == "enc-rs_1"
        assert final.text == "Sunny, 24C in Lisbon."


class TestStream:
    TOOL_EVENTS = (
        {"type": "response.created"},
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_1",
            "summary_index": 0,
            "delta": "Two calls",
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_1",
            "summary_index": 0,
            "delta": " needed.",
        },
        {"type": "response.output_text.delta", "item_id": "m1", "delta": "Checking "},
        {"type": "response.output_text.delta", "item_id": "m1", "delta": "both."},
        {"type": "response.output_text.delta", "item_id": "m2", "delta": "Done."},
        # Parallel calls interleave: items finish in any order.
        {
            "type": "response.output_item.done",
            "item": SimpleNamespace(**_function_call("c2", "get_weather", {"city": "Porto"})),
        },
        {
            "type": "response.output_item.done",
            "item": SimpleNamespace(**_function_call("c1", "get_weather", {"city": "Lisbon"})),
        },
    )

    def _final(self) -> SDKResponse:
        return _sdk_response(
            _reasoning("rs_1", ("Two calls needed.",)),
            _message("Checking both.", item_id="m1", phase="commentary"),
            _message("Done.", item_id="m2"),
            _function_call("c1", "get_weather", {"city": "Lisbon"}),
            _function_call("c2", "get_weather", {"city": "Porto"}),
        )

    async def test_stream_events(self):
        final = self._final()
        provider = _provider(_events(*self.TOOL_EVENTS, _completed(final)))

        events, response = await stream(provider, [USER], tools=[WEATHER_TOOL], thinking=True)

        texts = [e.text for e in events if e.kind == "text"]
        assert "".join(texts) == "Checking both.\n\nDone."
        thinking = [e for e in events if e.kind == "thinking"]
        assert [e.thinking.text for e in thinking] == ["Two calls", " needed."]
        assert all(e.partial for e in thinking)
        # Tool-call events come from the finished turn, once each, in output order — the order
        # the replayed calls and their results share (they used to follow arrival order).
        calls = [e.tool_call.id for e in events if e.kind == "tool_call"]
        assert calls == ["c1", "c2"]
        assert [tc.id for tc in response.tool_calls] == ["c1", "c2"]
        assert response.thinking == (ThinkingBlock(text="Two calls needed."),)
        assert response.usage == Usage(input_tokens=440, output_tokens=113, cache_read_tokens=113)
        assert response.stop_reason == "completed"
        assert response.raw is final

    async def test_a_streamed_parallel_tool_turn_replays_its_reasoning(self):
        client = _events(*self.TOOL_EVENTS, _completed(self._final()))
        async with LLM(MODEL, api_key="test-key") as llm:
            llm._provider._client = client  # type: ignore[attr-defined]
            stream = llm.stream_events([USER], tools=[WEATHER_TOOL])
            _ = [event async for event in stream]
            turn = stream.response

        assert turn is not None
        items = _input_items([USER, turn.to_message()])
        assert [item.get("type", "user") for item in items] == [
            "user",
            "reasoning",
            "message",
            "message",
            "function_call",
            "function_call",
        ]

    async def test_a_streamed_refusal_keeps_its_text_and_its_turn(self):
        refusal = {
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "refusal", "refusal": "I can't help with that."}],
        }
        client = _events(
            {"type": "response.refusal.delta", "item_id": "m1", "delta": "I can't "},
            {"type": "response.refusal.delta", "item_id": "m1", "delta": "help with that."},
            _completed(_sdk_response(_reasoning(), refusal)),
        )
        async with LLM(MODEL, api_key="test-key") as llm:
            llm._provider._client = client  # type: ignore[attr-defined]
            stream = llm.stream([USER])
            text = "".join([chunk async for chunk in stream])
            turn = stream.response

        assert text == "I can't help with that."
        assert turn is not None and turn.stop_reason == "refusal"
        assert [item.get("type", "user") for item in _input_items([USER, turn.to_message()])] == [
            "user",
            "reasoning",
            "message",
        ]

    async def test_stream_yields_text_only(self):
        provider = _provider(_events(*self.TOOL_EVENTS, _completed(self._final())))
        events, _ = await stream(provider, [USER])
        texts = [event.text for event in events if event.kind == "text"]
        assert texts == ["Checking ", "both.", "\n\n", "Done."]

    async def test_tool_calls_only_in_the_terminal_event_are_still_emitted(self):
        final = _sdk_response(_function_call("c1", "get_weather", {"city": "Lisbon"}))
        provider = _provider(_events(_completed(final)))

        events, response = await stream(provider, [USER])

        assert [e.tool_call.id for e in events if e.kind == "tool_call"] == ["c1"]
        assert [tc.id for tc in response.tool_calls] == ["c1"]

    async def test_incomplete_stream(self):
        final = _sdk_response(_reasoning(), status="incomplete", incomplete="max_output_tokens")
        provider = _provider(_events(_completed(final, "response.incomplete")))

        _, response = await stream(provider, [USER])

        assert response.stop_reason == "max_output_tokens"

    @pytest.mark.parametrize(
        ("event", "error", "status"),
        [
            (
                {"type": "error", "code": "server_shutting_down", "message": "draining"},
                APIError,
                503,
            ),
            (
                {"type": "error", "code": "rate_limit_exceeded", "message": "slow"},
                RateLimitError,
                429,
            ),
            # A 400 and a 500 both come with no code: no status can be told
            # (https://dev.meta.ai/docs/error-handling).
            ({"type": "error", "code": None, "message": "unknown"}, ResponseError, None),
            ({"type": "response.failed", "response": _failed("server_error")}, APIError, 500),
            # A code outside Meta's table has no documented status either.
            (
                {"type": "response.failed", "response": _failed("invalid_prompt")},
                ResponseError,
                None,
            ),
        ],
    )
    async def test_stream_errors(self, event, error, status):
        provider = _provider(_events({"type": "response.created"}, event))
        with pytest.raises(error) as caught:
            await stream(provider, [USER])
        assert getattr(caught.value, "status_code", None) == status

    async def test_a_failed_stream_keeps_the_usage_it_reported(self):
        event = {"type": "response.failed", "response": _failed("service_overloaded")}
        with pytest.raises(APIError) as caught:
            await stream(_provider(_events(event)), [USER])
        assert caught.value.status_code == 503
        assert caught.value.usage == Usage(input_tokens=100, output_tokens=40)

    async def test_a_deterministic_stream_failure_is_not_retried(self):
        async def _failing() -> Any:
            yield SimpleNamespace(type="response.failed", response=_failed("invalid_prompt"))

        client = AsyncMock()
        client.responses.create = AsyncMock(side_effect=lambda **_: OpenAIStream(_failing()))
        async with LLM(MODEL, api_key="test-key", retry=RetryConfig(base_delay=0.01)) as llm:
            llm._provider._client = client  # type: ignore[attr-defined]
            with pytest.raises(ResponseError):
                _ = [chunk async for chunk in llm.stream([USER])]

        assert client.responses.create.await_count == 1

    async def test_an_sdk_error_payload_mid_stream_is_mapped(self):
        # The SDK raises a bare openai.APIError for an event whose data carries an "error" key.
        async def _broken() -> Any:
            yield SimpleNamespace(type="response.created")
            raise openai.APIError(
                "The backend is temporarily overloaded.",
                request=httpx.Request("POST", f"{DEFAULT_BASE_URL}/responses"),
                body={"code": "service_overloaded", "message": "overloaded"},
            )

        client = AsyncMock()
        client.responses.create = AsyncMock(return_value=OpenAIStream(_broken()))
        with pytest.raises(APIError) as caught:
            await stream(_provider(client), [USER])
        assert caught.value.status_code == 503

    async def test_status_error_when_opening_the_stream(self):
        client = AsyncMock()
        client.responses.create.side_effect = _status_error(openai.BadRequestError, 400)
        with pytest.raises(APIError) as caught:
            await stream(_provider(client), [USER])
        assert caught.value.status_code == 400


class TestNetworkErrors:
    def test_a_connection_that_never_opened_was_not_sent(self):
        request = httpx.Request("POST", f"{DEFAULT_BASE_URL}/responses")
        refused = openai.APIConnectionError(request=request)
        refused.__cause__ = httpx2.ConnectError("refused")
        error = _provider().map_error(refused, sent=True)
        assert (type(error), error.delivery) == (TransportError, "not_sent")
        slow = _provider().map_error(httpx2.ReadTimeout("slow"), sent=True)
        assert (type(slow), slow.delivery) == (ProviderTimeout, "indeterminate")

    @pytest.mark.parametrize(
        ("sdk_error", "expected"),
        [(openai.APIConnectionError, ConnectionError), (openai.APITimeoutError, TimeoutError)],
    )
    async def test_network_failures_become_builtin_errors(self, sdk_error, expected):
        error = sdk_error(request=httpx.Request("POST", f"{DEFAULT_BASE_URL}/responses"))
        client = AsyncMock()
        client.responses.create.side_effect = error
        client.responses.input_tokens.count = AsyncMock(side_effect=error)
        provider = _provider(client)

        with pytest.raises(expected):
            await complete(provider, [USER])
        with pytest.raises(expected):
            await provider.count_tokens([USER])
        with pytest.raises(expected):
            await stream(provider, [USER])


class TestCountTokens:
    async def test_counts_with_the_input_tokens_endpoint(self):
        client = AsyncMock()
        client.responses.input_tokens.count = AsyncMock(
            return_value=SimpleNamespace(input_tokens=685)
        )

        count = await _provider(client).count_tokens(
            [USER],
            system="Be brief.",
            tools=[WEATHER_TOOL, {"_server_tool": True, "type": "web_search"}],
        )

        assert count == 685
        sent = client.responses.input_tokens.count.call_args.kwargs
        assert sent["instructions"] == "Be brief."
        assert [tool["name"] for tool in sent["tools"]] == ["get_weather"]


class TestClient:
    def test_defaults_to_the_meta_endpoint_without_sdk_retries(self):
        client = MetaProvider(MODEL, "test-key")._client
        assert str(client.base_url).rstrip("/") == DEFAULT_BASE_URL
        assert client.max_retries == 0

    def test_base_url_override(self):
        client = MetaProvider(MODEL, "test-key", base_url="https://gateway.example/v1")._client
        assert str(client.base_url).rstrip("/") == "https://gateway.example/v1"

    def test_openai_account_headers_never_reach_meta(self, monkeypatch):
        monkeypatch.setenv("OPENAI_ORG_ID", "org-secret")
        monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-secret")
        client = MetaProvider(MODEL, "test-key")._client
        sent = {k for k, v in client.default_headers.items() if isinstance(v, str)}
        assert not {"OpenAI-Organization", "OpenAI-Project"} & sent
        assert client.auth_headers == {"Authorization": "Bearer test-key"}

    def test_the_client_marks_the_dispatch_and_takes_a_plain_timeout(self, monkeypatch):
        # The meta extra installs openai, whose transport is httpx2: httpx may be absent.
        monkeypatch.setitem(sys.modules, "httpx", None)
        client = MetaProvider(MODEL, "test-key", timeout=5.0)._client
        assert client.timeout == 5.0
        assert client._client.event_hooks["request"] == [on_request]

    async def test_close(self):
        client = AsyncMock()
        await _provider(client).close()
        client.close.assert_awaited_once()
