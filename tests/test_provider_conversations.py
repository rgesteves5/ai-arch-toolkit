"""Canonical neutral histories on each provider's wire (R02, the conversation contract).

The histories are the ones an agent produces: one call; two parallel calls and their results;
text with calls; thinking with calls; and a turn that came from a stream. Each adapter must
convert them by its provider's documented sequencing rules: a turn's results together and in the
order of the calls, with the call id where the provider has one.
"""

from __future__ import annotations

from typing import Any

import pytest
from anthropic import types as claude
from google.genai import types as gemini
from xai_sdk import chat as xai_chat
from xai_sdk.proto import chat_pb2

from ai_arch_toolkit.core import Response, ThinkingBlock, ToolCall, tool_result, user
from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider
from ai_arch_toolkit.core._providers._gemini import GeminiProvider
from ai_arch_toolkit.core._providers._meta import MetaProvider
from ai_arch_toolkit.core._providers._openai import OpenAIProvider
from ai_arch_toolkit.core._providers._xai import XAIProvider
from tests.integration import fakegrpc
from tests.provider_calls import prepare

WEATHER = ToolCall(id="call_1", name="get_weather", input={"city": "Lisbon"})
TIME = ToolCall(id="call_2", name="get_time", input={"tz": "UTC"})
QUESTION = user("Weather and time in Lisbon?")


def _turn(text: str, *calls: ToolCall, thinking: str = "") -> dict[str, Any]:
    thought = (ThinkingBlock(text=thinking),) if thinking else ()
    return Response(text=text, tool_calls=calls, thinking=thought).to_message()


def _results(*calls: ToolCall) -> list[dict[str, Any]]:
    answers = {"get_weather": "Sunny, 24C", "get_time": "14:05"}
    return [tool_result(answers[c.name], tool_use_id=c.id, name=c.name) for c in calls]


HISTORIES: dict[str, list[dict[str, Any]]] = {
    "one call": [QUESTION, _turn("", WEATHER), *_results(WEATHER)],
    "two parallel calls": [QUESTION, _turn("", WEATHER, TIME), *_results(WEATHER, TIME)],
    "text with calls": [
        QUESTION,
        _turn("Checking both.", WEATHER, TIME),
        *_results(WEATHER, TIME),
    ],
    "thinking with calls": [
        QUESTION,
        _turn("", WEATHER, TIME, thinking="Two lookups."),
        *_results(WEATHER, TIME),
    ],
}


# ---------------------------------------------------------------------------
# OpenAI Chat Completions: the assistant message carries every call of the turn; each result is
# its own `tool` message with the call's id, in call order. Reasoning is not replayed.
# ---------------------------------------------------------------------------


def _openai_wire(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(prepare(OpenAIProvider("gpt-5.4", "test-key"), history).params["messages"])


@pytest.mark.parametrize("name", HISTORIES)
def test_openai_keeps_a_turn_whole_and_its_results_in_call_order(name: str) -> None:
    wire = _openai_wire(HISTORIES[name])
    assistant = wire[1]
    calls = [call for message in HISTORIES[name][1:2] for call in message["tool_calls"]]
    assert assistant["role"] == "assistant"
    assert [c["id"] for c in assistant["tool_calls"]] == [c["id"] for c in calls]
    assert all(c["type"] == "function" for c in assistant["tool_calls"])
    results = wire[2:]
    assert [(r["role"], r["tool_call_id"]) for r in results] == [("tool", c["id"]) for c in calls]
    assert "reasoning" not in str(assistant)  # Chat Completions replays no reasoning


def test_openai_text_that_leads_into_calls_is_kept() -> None:
    wire = _openai_wire(HISTORIES["text with calls"])
    assert wire[1]["content"] == "Checking both."


def test_openai_a_streamed_turn_replays_like_a_completed_one() -> None:
    from tests.sdk_streams import openai_chunk

    # to_message() of a streamed response carries the SDK's snapshot as _raw; the wire must
    # not depend on it.
    snapshot_raw = openai_chunk(content="x")
    streamed = {**HISTORIES["two parallel calls"][1], "_raw": snapshot_raw}
    history = [QUESTION, streamed, *_results(WEATHER, TIME)]
    assert _openai_wire(history) == _openai_wire(HISTORIES["two parallel calls"])


# ---------------------------------------------------------------------------
# xAI (gRPC): the assistant message carries every call of the turn; each result is its own tool
# message with the call's id, in call order (https://docs.x.ai/docs/guides/function-calling).
# Reasoning is not replayed: the adapter asks for no encrypted reasoning.
# ---------------------------------------------------------------------------

ROLE = chat_pb2.MessageRole


async def _xai_wire(history: list[dict[str, Any]]) -> list[Any]:
    async with XAIProvider("grok-4.6", "test-key") as provider:
        return list(prepare(provider, history).params.proto.messages)


@pytest.mark.parametrize("name", HISTORIES)
async def test_xai_keeps_a_turn_whole_and_its_results_in_call_order(name: str) -> None:
    wire = await _xai_wire(HISTORIES[name])
    assistant = wire[1]
    calls = [call for message in HISTORIES[name][1:2] for call in message["tool_calls"]]
    assert assistant.role == ROLE.ROLE_ASSISTANT
    assert [(c.id, c.function.name) for c in assistant.tool_calls] == [
        (c["id"], c["name"]) for c in calls
    ]
    results = wire[2:]
    assert [(r.role, r.tool_call_id) for r in results] == [
        (ROLE.ROLE_TOOL, c["id"]) for c in calls
    ]
    assert not assistant.reasoning_content


async def test_xai_text_that_leads_into_calls_is_kept() -> None:
    wire = await _xai_wire(HISTORIES["text with calls"])
    assert [part.text for part in wire[1].content] == ["Checking both."]


async def test_xai_a_streamed_turn_replays_like_a_completed_one() -> None:
    # to_message() of a streamed response carries the SDK's accumulated response as _raw; the
    # wire must not depend on it.
    snapshot_raw = xai_chat.Response(fakegrpc.answer("x"), 0)
    streamed = {**HISTORIES["two parallel calls"][1], "_raw": snapshot_raw}
    history = [QUESTION, streamed, *_results(WEATHER, TIME)]
    assert await _xai_wire(history) == await _xai_wire(HISTORIES["two parallel calls"])


# ---------------------------------------------------------------------------
# Gemini: the model turn keeps every part and thought signature; a turn's results go back together
# in one user content, in call order, each with the call's id when Gemini gave one
# (https://ai.google.dev/gemini-api/docs/generate-content/function-calling).
# ---------------------------------------------------------------------------


def _gemini_wire(history: list[dict[str, Any]]) -> list[Any]:
    return list(
        prepare(GeminiProvider("gemini-3.8-flash", "test-key"), history).params["contents"]
    )


@pytest.mark.parametrize("name", HISTORIES)
def test_gemini_keeps_a_turn_whole_and_its_results_together(name: str) -> None:
    wire = _gemini_wire(HISTORIES[name])
    calls = [call for message in HISTORIES[name][1:2] for call in message["tool_calls"]]
    assert [content.role for content in wire] == ["user", "model", "user"]
    sent = [part.function_call.name for part in wire[1].parts if part.function_call]
    assert sent == [call["name"] for call in calls]
    responses = [part.function_response for part in wire[2].parts]
    assert [r.name for r in responses] == [call["name"] for call in calls]
    # The toolkit's ids were never Gemini's: none goes back.
    assert [r.id for r in responses] == [None] * len(calls)


def _gemini_turn() -> dict[str, Any]:
    """A Gemini 3 turn with two calls: the ids and the signature are Gemini's."""
    raw = gemini.GenerateContentResponse(
        candidates=[
            gemini.Candidate(
                content=gemini.Content(
                    role="model",
                    parts=[
                        gemini.Part(text="Two lookups.", thought=True, thought_signature=b"t"),
                        gemini.Part(
                            function_call=gemini.FunctionCall(
                                id="call_1", name="get_weather", args={"city": "Lisbon"}
                            ),
                            thought_signature=b"sig",
                        ),
                        gemini.Part(
                            function_call=gemini.FunctionCall(
                                id="call_2", name="get_time", args={"tz": "UTC"}
                            )
                        ),
                    ],
                )
            )
        ]
    )
    return {**_turn("", WEATHER, TIME, thinking="Two lookups."), "_raw": raw}


def test_gemini_replays_its_own_turn_and_returns_its_ids() -> None:
    wire = _gemini_wire([QUESTION, _gemini_turn(), *_results(WEATHER, TIME)])
    assert [part.thought_signature for part in wire[1].parts] == [b"t", b"sig", None]
    responses = [part.function_response for part in wire[2].parts]
    assert [(r.id, r.name) for r in responses] == [
        ("call_1", "get_weather"),
        ("call_2", "get_time"),
    ]


def test_gemini_text_that_leads_into_calls_is_kept() -> None:
    wire = _gemini_wire(HISTORIES["text with calls"])
    assert wire[1].parts[0].text == "Checking both."


# ---------------------------------------------------------------------------
# Meta (Responses API): each call is its own function_call item and each result a
# function_call_output with the call's id, in call order; text that leads into calls is a
# commentary message. Reasoning replays only from the response kept in _raw (D12).
# ---------------------------------------------------------------------------


def _meta_wire(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(prepare(MetaProvider("muse-spark-1.3", "test-key"), history).params["input"])


@pytest.mark.parametrize("name", HISTORIES)
def test_meta_keeps_a_turn_whole_and_its_results_in_call_order(name: str) -> None:
    wire = _meta_wire(HISTORIES[name])
    calls = [call for message in HISTORIES[name][1:2] for call in message["tool_calls"]]
    sent = [item for item in wire if item.get("type") == "function_call"]
    assert [(c["call_id"], c["name"]) for c in sent] == [(c["id"], c["name"]) for c in calls]
    results = [item for item in wire if item.get("type") == "function_call_output"]
    assert [r["call_id"] for r in results] == [c["id"] for c in calls]
    assert wire.index(sent[-1]) < wire.index(results[0])  # the turn, then its results
    assert not [item for item in wire if item.get("type") == "reasoning"]


def test_meta_text_that_leads_into_calls_is_commentary() -> None:
    wire = _meta_wire(HISTORIES["text with calls"])
    message = next(item for item in wire if item.get("type") == "message")
    assert message["phase"] == "commentary"
    assert message["content"] == [{"type": "output_text", "text": "Checking both."}]


# ---------------------------------------------------------------------------
# Anthropic: every tool_result of a turn goes back in one user message, in call order and before
# any text (https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use); the
# assistant turn is replayed as Claude sent it, thinking blocks and signatures included, while it
# still matches the message (https://platform.claude.com/docs/en/build-with-claude/thinking).
# ---------------------------------------------------------------------------


def _claude_wire(history: list[dict[str, Any]]) -> list[Any]:
    provider = AnthropicProvider("claude-opus-5", "test-key")
    return list(prepare(provider, history, max_tokens=1024).params["messages"])


def _block(block: Any) -> dict[str, Any]:
    return block if isinstance(block, dict) else block.model_dump(exclude_none=True)


@pytest.mark.parametrize("name", HISTORIES)
def test_claude_sends_a_turns_results_in_one_user_message(name: str) -> None:
    wire = _claude_wire(HISTORIES[name])
    calls = [call for message in HISTORIES[name][1:2] for call in message["tool_calls"]]
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    uses = [_block(b) for b in wire[1]["content"] if _block(b)["type"] == "tool_use"]
    assert [(u["id"], u["name"]) for u in uses] == [(c["id"], c["name"]) for c in calls]
    results = [_block(b) for b in wire[2]["content"]]
    assert [(r["type"], r["tool_use_id"]) for r in results] == [
        ("tool_result", c["id"]) for c in calls
    ]


def _claude_turn(text: str = "Checking both.") -> dict[str, Any]:
    """A turn Claude sent: its thinking block and signature ride along in _raw."""
    content: list[Any] = [
        claude.ThinkingBlock(type="thinking", thinking="Two lookups.", signature="sig")
    ]
    if text:
        content.append(claude.TextBlock(type="text", text=text))
    content += [
        claude.ToolUseBlock(type="tool_use", id=c.id, name=c.name, input=c.input)
        for c in (WEATHER, TIME)
    ]
    raw = claude.Message(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-opus-5",
        content=content,
        stop_reason="tool_use",
        usage=claude.Usage(input_tokens=10, output_tokens=5),
    )
    return {**_turn(text, WEATHER, TIME, thinking="Two lookups."), "_raw": raw}


def test_claude_replays_its_own_turn_with_the_thinking_signature() -> None:
    wire = _claude_wire([QUESTION, _claude_turn(), *_results(WEATHER, TIME)])
    blocks = [_block(b) for b in wire[1]["content"]]
    assert blocks[0] == {"type": "thinking", "thinking": "Two lookups.", "signature": "sig"}
    assert [b["type"] for b in blocks] == ["thinking", "text", "tool_use", "tool_use"]


def test_claude_rebuilds_a_turn_whose_text_was_edited() -> None:
    edited = {**_claude_turn(), "content": "Something else."}
    wire = _claude_wire([QUESTION, edited, *_results(WEATHER, TIME)])
    blocks = [_block(b) for b in wire[1]["content"]]
    assert [b["type"] for b in blocks] == ["text", "tool_use", "tool_use"]
    assert blocks[0]["text"] == "Something else."
