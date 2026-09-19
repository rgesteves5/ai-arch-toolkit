"""Canaries for the wire net (R02 step 4): known-bad requests are caught, good ones pass.

``tests/wire_contract.py`` checks each request an adapter's ``prepare`` builds against its SDK's
own request contract, and the autouse ``wire_log`` fixture (``tests/conftest.py``) runs it on
every request the suite builds. These tests prove both ends: the net catches what it is for, and
lets through what the five adapters build for a full request.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

import pytest
from google.genai import types as gemini

from ai_arch_toolkit.core import Response, ThinkingBlock, ToolCall, _providers, tool_result, user
from ai_arch_toolkit.core._content import document, image, system
from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider
from ai_arch_toolkit.core._providers._base import BaseProvider
from ai_arch_toolkit.core._providers._gemini import GeminiProvider
from ai_arch_toolkit.core._providers._meta import MetaProvider
from ai_arch_toolkit.core._providers._openai import OpenAIProvider
from ai_arch_toolkit.core._providers._xai import XAIProvider
from ai_arch_toolkit.core._response import OutputSchema
from tests.provider_calls import prepare
from tests.wire_contract import ADAPTERS, VALIDATORS, WireLog, violations

WEATHER = {
    "name": "get_weather",
    "description": "Get the weather",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}
CALLS = (
    ToolCall(id="call_1", name="get_weather", input={"city": "Lisbon"}),
    ToolCall(id="call_2", name="get_weather", input={"city": "Porto"}),
)
# One of everything a request carries: system, text, an image, a document, a turn with text,
# thinking and two parallel calls, and their results.
HISTORY = [
    system("Be brief."),
    user(["Compare these:", image(b"\x89PNG"), document(b"%PDF-1.4", name="a.pdf")]),
    Response(
        text="Checking both.", tool_calls=CALLS, thinking=(ThinkingBlock(text="Two lookups."),)
    ).to_message(),
    *(tool_result("Sunny", tool_use_id=call.id, name=call.name) for call in CALLS),
]
SCHEMA = OutputSchema(
    name="Answer",
    schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
)


def _full(provider: Any, **kwargs: Any) -> Any:
    """What ``provider`` builds for the full history, with tools and structured output."""
    return prepare(
        provider, HISTORY, tools=[WEATHER], max_tokens=1024, output_schema=SCHEMA, **kwargs
    ).params


def _only(found: list[str], *fragments: str) -> None:
    """The net reports each violation the canary planted."""
    assert found, "the net let a known-bad request through"
    for fragment in fragments:
        assert any(fragment in line for line in found), (fragment, found)


# ---------------------------------------------------------------------------
# OpenAI: CompletionCreateParamsNonStreaming
# ---------------------------------------------------------------------------

OPENAI_OK = {"model": "gpt-5-nano", "messages": [{"role": "user", "content": "hi"}]}


class TestOpenAI:
    def test_a_full_request_conforms(self) -> None:
        params = _full(OpenAIProvider("gpt-5.4", "k"), thinking_effort="low")

        assert violations("OpenAIProvider", params) == []

    def test_a_compatible_server_request_conforms(self) -> None:
        provider = OpenAIProvider("llama3", "k", base_url="http://127.0.0.1:11434/v1")

        assert violations("OpenAIProvider", _full(provider)) == []

    @pytest.mark.parametrize(
        ("change", "planted"),
        [
            ({"max_output_tokens": 5}, "max_output_tokens | Extra inputs are not permitted"),
            ({"temperature": "hot"}, "temperature |"),
            ({"reasoning_effort": "extreme"}, "reasoning_effort |"),
            ({"messages": [{"role": "robot", "content": "hi"}]}, "messages.0 |"),
            ({"messages": [{"role": "tool", "content": "x"}]}, "tool_call_id | Field required"),
            (
                {"tools": [{"type": "function", "name": "f", "parameters": {}}]},
                "tools.0.function.function | Field required",
            ),
            # Chat Completions runs no server tools (the finding "Server tools no fio").
            ({"tools": [{"type": "web_search"}]}, "tools.0 |"),
        ],
    )
    def test_a_known_bad_request_is_caught(self, change: dict[str, Any], planted: str) -> None:
        _only(violations("OpenAIProvider", {**OPENAI_OK, **change}), planted)


# ---------------------------------------------------------------------------
# Anthropic: MessageCreateParamsNonStreaming, sampling in extra_body
# ---------------------------------------------------------------------------

CLAUDE_OK = {
    "model": "claude-haiku-4-5",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hi"}],
}


class TestAnthropic:
    @pytest.mark.parametrize(
        ("model", "kwargs"),
        [
            ("claude-opus-5", {"thinking": True, "thinking_effort": "high"}),
            ("claude-sonnet-4-6", {"temperature": 0.2, "top_k": 5}),
            ("claude-haiku-4-5", {"thinking": True, "thinking_budget": 2048}),
        ],
    )
    def test_a_full_request_conforms(self, model: str, kwargs: dict[str, Any]) -> None:
        params = _full(AnthropicProvider(model, "k"), **kwargs)

        assert violations("AnthropicProvider", params) == []

    @pytest.mark.parametrize(
        ("change", "planted"),
        [
            # anthropic 1.x took the sampling fields out of the params: at the top level they
            # are unknown to the SDK (the b3dae3f fix sends them in the body instead).
            ({"temperature": 0.2}, "temperature | Extra inputs are not permitted"),
            ({"extra_body": {"frequency_penalty": 0.5}}, "extra_body.frequency_penalty |"),
            ({"extra_body": {"top_k": "many"}}, "extra_body.top_k |"),
            (
                {"thinking": {"type": "adaptive", "budget_tokens": 1024}},
                "thinking.adaptive.budget_tokens | Extra inputs are not permitted",
            ),
            ({"tools": [{"type": "web_search_20250305"}]}, "name | Field required"),
            (
                {"messages": [{"role": "user", "content": [{"type": "tool_result"}]}]},
                "tool_use_id | Field required",
            ),
        ],
    )
    def test_a_known_bad_request_is_caught(self, change: dict[str, Any], planted: str) -> None:
        _only(violations("AnthropicProvider", {**CLAUDE_OK, **change}), planted)


# ---------------------------------------------------------------------------
# Meta: ResponseCreateParamsNonStreaming, with the adapter's three documented deviations
# ---------------------------------------------------------------------------

META_OK = {"model": "muse-spark-1.3", "store": False, "input": "hi"}


class TestMeta:
    def test_a_full_request_conforms(self) -> None:
        params = _full(MetaProvider("muse-spark-1.3", "k"), thinking_effort="low")

        assert violations("MetaProvider", params) == []

    def test_the_adapter_still_makes_each_documented_deviation(self) -> None:
        """The net fills in exactly these fields; once the adapter stops leaving one out, the
        fill for it goes (``tests/wire_contract.py``)."""
        params = _full(MetaProvider("muse-spark-1.3", "k"))
        items = list(params["input"])
        rebuilt = next(i for i in items if i.get("role") == "assistant")
        picture = next(p for p in items[1]["content"] if p["type"] == "input_image")

        assert "id" not in rebuilt and "status" not in rebuilt
        assert "annotations" not in rebuilt["content"][0]
        assert "strict" not in params["tools"][0]
        assert "detail" not in picture

    @pytest.mark.parametrize(
        ("change", "planted"),
        [
            ({"max_tokens": 5}, "max_tokens | Extra inputs are not permitted"),
            ({"reasoning": {"effort": "extreme"}}, "reasoning.effort |"),
            ({"input": [{"type": "function_call_output", "call_id": "c1"}]}, "output |"),
            # The fill never hides a wrong value in a deviation's own field.
            (
                {"tools": [{"type": "function", "name": "f", "parameters": {}, "strict": "yes"}]},
                "strict |",
            ),
            (
                {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_image", "image_url": "u", "detail": "ultra"}
                            ],
                        }
                    ]
                },
                "detail |",
            ),
        ],
    )
    def test_a_known_bad_request_is_caught(self, change: dict[str, Any], planted: str) -> None:
        _only(violations("MetaProvider", {**META_OK, **change}), planted)


# ---------------------------------------------------------------------------
# Gemini: the SDK's own validation and its offline Developer API conversion
# ---------------------------------------------------------------------------


def _gemini_request(**config: Any) -> dict[str, Any]:
    return {
        "model": "gemini-3.1-flash-lite",
        "contents": [gemini.Content(role="user", parts=[gemini.Part(text="hi")])],
        "config": gemini.GenerateContentConfig(**config),
    }


class TestGemini:
    @pytest.mark.parametrize(
        ("model", "kwargs"),
        [
            ("gemini-3.1-flash-lite", {"thinking": True, "thinking_effort": "low"}),
            ("gemini-2.5-flash", {"thinking_budget": 2048, "temperature": 0.2}),
        ],
    )
    def test_a_full_request_conforms(self, model: str, kwargs: dict[str, Any]) -> None:
        params = _full(GeminiProvider(model, "k"), **kwargs)

        assert violations("GeminiProvider", params) == []

    def test_a_field_only_vertex_takes_is_caught(self) -> None:
        request = _gemini_request()
        blob = gemini.Blob(data=b"x", mime_type="image/png", display_name="a.png")
        request["contents"] = [gemini.Content(role="user", parts=[gemini.Part(inline_data=blob)])]

        _only(violations("GeminiProvider", request), "display_name")

    def test_a_value_set_after_construction_is_validated(self) -> None:
        request = _gemini_request()
        request["config"].temperature = "hot"  # pydantic does not validate assignment

        _only(violations("GeminiProvider", request), "temperature")

    def test_an_enum_value_the_sdk_only_warns_about_is_caught(self) -> None:
        thinking = gemini.ThinkingConfig.model_construct(thinking_level="ULTRA")

        _only(violations("GeminiProvider", _gemini_request(thinking_config=thinking)), "ULTRA")


# ---------------------------------------------------------------------------
# xAI: the SDK's Chat, and its proto's enums
# ---------------------------------------------------------------------------


class TestXai:
    @pytest.mark.filterwarnings("ignore:xAI does not support")  # its images and documents
    async def test_a_full_request_conforms(self) -> None:
        async with XAIProvider("grok-4.3", "k") as provider:
            params = _full(provider, thinking_effort="low")

        assert violations("XAIProvider", params) == []

    async def test_an_enum_value_the_proto_does_not_define_is_caught(self) -> None:
        async with XAIProvider("grok-4.3", "k") as provider:
            chat = prepare(provider, [user("hi")], max_tokens=16).params
        chat.proto.reasoning_effort = 99
        chat.proto.messages[0].role = 42

        _only(
            violations("XAIProvider", chat),
            "reasoning_effort | 99 is not a ReasoningEffort",
            "messages.0.role | 42 is not a MessageRole",
        )

    def test_a_request_that_is_not_the_sdks_chat_is_caught(self) -> None:
        _only(violations("XAIProvider", {"model": "grok-4.3"}), "not an xai_sdk Chat")


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------


def _adapters() -> set[str]:
    """Every adapter class defined in ``core/_providers``."""
    found: set[str] = set()
    for module in pkgutil.iter_modules(_providers.__path__):
        loaded = importlib.import_module(f"{_providers.__name__}.{module.name}")
        found.update(
            name
            for name, cls in inspect.getmembers(loaded, inspect.isclass)
            if issubclass(cls, BaseProvider)
            and cls is not BaseProvider
            and cls.__module__ == loaded.__name__
        )
    return found


class TestFixture:
    def test_every_adapter_is_under_the_net(self) -> None:
        assert {a.__name__ for a in ADAPTERS} == set(VALIDATORS) == _adapters()

    def test_every_prepared_request_is_checked(self, wire_log: WireLog) -> None:
        prepare(MetaProvider("muse-spark-1.3", "k"), [user("hi")])
        prepare(OpenAIProvider("gpt-5-nano", "k"), [user("hi")])

        assert wire_log.checked == ["MetaProvider", "OpenAIProvider"]

    def test_a_violation_fails_unless_the_test_declares_it(self) -> None:
        log = WireLog()
        log.record("OpenAIProvider", {**OPENAI_OK, "temperature": "hot"})

        assert log.unexpected(()) == log.violations
        assert log.violations[0].startswith("OpenAIProvider: temperature |")
        assert log.unexpected([r"^OpenAIProvider: temperature \|"]) == []
