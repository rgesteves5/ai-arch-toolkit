"""Capture the real kwargs each adapter hands to its SDK, through the public LLM API.

Nothing here touches the network: every SDK client is replaced by a mock, exactly as the repo's
unit tests do (``llm._provider._client = AsyncMock()``).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel, Field

from ai_arch_toolkit.core import (
    LLM,
    Response,
    ToolCall,
    cache,
    code_execution,
    document,
    image,
    tool,
    tool_result,
    user,
    web_search,
)

# --------------------------------------------------------------------------------------
# Shared scenario inputs
# --------------------------------------------------------------------------------------


class Address(BaseModel):
    """Where the person lives."""

    city: str
    zip_code: str | None = None


class Person(BaseModel):
    name: str
    address: Address
    tags: list[str] = Field(default_factory=list)


class Node(BaseModel):
    label: str
    children: list[Node] = Field(default_factory=list)


class Verdict(BaseModel):
    """Structured-output model with a nested model, an optional and a default."""

    person: Person
    score: float
    note: str | None = None
    labels: list[str] = Field(default_factory=list)


@tool
def save_person(person: Person, notify: bool = False) -> str:
    """Save a person (nested pydantic model -> inlined schema)."""
    return "ok"


@tool
def store_tree(root: Node) -> str:
    """Store a tree (recursive model -> $defs hoisted to the tool root)."""
    return "ok"


@tool
def get_weather(city: str, units: str = "c") -> str:
    """Get the weather for a city."""
    return "sunny"


RAW_DEFS_TOOL: dict[str, Any] = {
    "name": "lookup_item",
    "description": "Look up an item.",
    "input_schema": {
        "type": "object",
        "properties": {"item": {"$ref": "#/$defs/Item"}},
        "$defs": {"Item": {"type": "object", "properties": {"name": {"type": "string"}}}},
        "required": ["item"],
    },
}

FN_TOOLS = [get_weather, save_person, store_tree, RAW_DEFS_TOOL]


def history_helpers() -> list[dict[str, Any]]:
    """Assistant tool call + two consecutive tool results, built with the library helpers."""
    turn = Response(
        text="Let me check both cities.",
        tool_calls=(
            ToolCall(id="call_1", name="get_weather", input={"city": "Lisbon"}),
            ToolCall(id="call_2", name="get_weather", input={"city": "Porto"}),
        ),
    )
    return [
        user("Weather in Lisbon and Porto?"),
        turn.to_message(),
        tool_result("sunny", tool_use_id="call_1", name="get_weather"),
        tool_result({"temp": 21, "sky": "cloudy"}, tool_use_id="call_2", name="get_weather"),
    ]


def history_str_results() -> list[dict[str, Any]]:
    """Same shape as the library's own runner produces: no assistant text, string results."""
    turn = Response(
        text="",
        tool_calls=(
            ToolCall(id="call_1", name="get_weather", input={"city": "Lisbon"}),
            ToolCall(id="call_2", name="get_weather", input={"city": "Porto"}),
        ),
    )
    return [
        user("Weather in Lisbon and Porto?"),
        turn.to_message(),
        tool_result("sunny", tool_use_id="call_1", name="get_weather"),
        tool_result("cloudy", tool_use_id="call_2", name="get_weather"),
    ]


def multimodal() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [cache("Long shared context."), "Be brief."]},
        user(
            [
                "Describe these.",
                image(b"\x89PNG\r\n"),
                image("https://example.com/cat.png"),
                document(b"%PDF-1.4", name="spec.pdf"),
                cache("A long document to cache."),
            ]
        ),
    ]


@dataclass(frozen=True, slots=True, kw_only=True)
class Scenario:
    name: str
    mode: str = "complete"  # complete | stream | stream_events
    messages: Any = "Hi"
    call: dict[str, Any] = field(default_factory=dict)


def scenarios() -> list[Scenario]:
    hist = history_helpers
    out = [
        Scenario(name="1-plain"),
        Scenario(name="1-plain+system", call={"system": "Be brief."}),
        Scenario(name="2-fn-tools", call={"tools": FN_TOOLS}),
        Scenario(name="3-choice-auto", call={"tools": FN_TOOLS, "tool_choice": "auto"}),
        Scenario(name="3-choice-required", call={"tools": FN_TOOLS, "tool_choice": "required"}),
        Scenario(name="3-choice-none", call={"tools": FN_TOOLS, "tool_choice": "none"}),
        Scenario(name="3-choice-named", call={"tools": FN_TOOLS, "tool_choice": "get_weather"}),
        Scenario(name="4-server-web_search", call={"tools": [web_search(max_uses=3)]}),
        Scenario(name="4-server-code_exec", call={"tools": [code_execution()]}),
        Scenario(
            name="4-server-both+fn",
            call={"tools": [get_weather, web_search(max_uses=3), code_execution()]},
        ),
        Scenario(name="5-thinking", call={"thinking": True}),
        Scenario(name="5-thinking+effort", call={"thinking": True, "thinking_effort": "low"}),
        Scenario(name="5-effort-only", call={"thinking_effort": "high"}),
        Scenario(name="6-output_schema", call={"output_schema": Verdict}),
        Scenario(name="6-json_mode", call={"json_mode": True}),
        Scenario(name="7-history-helpers", messages=hist, call={"tools": [get_weather]}),
        Scenario(name="7-history-str-results", messages=history_str_results, call={"tools": [get_weather]}),
        Scenario(name="7-history-raw-replay", messages="__replay__", call={"tools": [get_weather]}),
        Scenario(name="9-multimodal(extra)", messages=multimodal),
        Scenario(name="8-stream-plain", mode="stream"),
        Scenario(name="8-stream-tools", mode="stream", call={"tools": FN_TOOLS}),
        Scenario(name="8-events-thinking", mode="stream_events", call={"thinking": True}),
        Scenario(name="8-events-schema", mode="stream_events", call={"output_schema": Verdict}),
    ]
    return out


# --------------------------------------------------------------------------------------
# Rigs: one per adapter. Each returns (llm, grab) where grab() -> list of captured kwargs.
# --------------------------------------------------------------------------------------


class _AnthropicStream:
    def __init__(self) -> None:
        self._final = SimpleNamespace(content=[])

    async def __aenter__(self) -> _AnthropicStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def __aiter__(self) -> _AnthropicStream:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration

    async def get_final_message(self) -> Any:
        return self._final


def _anthropic_message(tool_calls: bool = False) -> Any:
    from anthropic import types as t

    content: list[Any] = [t.TextBlock(type="text", text="Let me check.")]
    if tool_calls:
        content += [
            t.ToolUseBlock(type="tool_use", id="toolu_1", name="get_weather", input={"city": "Lisbon"}),
            t.ToolUseBlock(type="tool_use", id="toolu_2", name="get_weather", input={"city": "Porto"}),
        ]
    return t.Message(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-sonnet-4-6",
        content=content,
        stop_reason="tool_use" if tool_calls else "end_turn",
        usage=t.Usage(input_tokens=10, output_tokens=5),
    )


def rig_anthropic(model: str = "claude-sonnet-4-6") -> tuple[LLM, Callable[[], list[tuple[str, dict]]], Callable[[], Any]]:
    llm = LLM(model, api_key="x")
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_anthropic_message())
    client.messages.stream = MagicMock(side_effect=lambda **_: _AnthropicStream())
    llm._provider._client = client  # type: ignore[attr-defined]

    def grab() -> list[tuple[str, dict]]:
        calls = [("messages.create", c.kwargs) for c in client.messages.create.call_args_list]
        calls += [("messages.stream", c.kwargs) for c in client.messages.stream.call_args_list]
        return calls

    def tool_turn() -> Any:
        client.messages.create.return_value = _anthropic_message(tool_calls=True)
        return client

    return llm, grab, tool_turn


def _openai_completion(tool_calls: bool = False) -> Any:
    from openai.types.chat import ChatCompletion

    message: dict[str, Any] = {"role": "assistant", "content": "Let me check."}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps({"city": c})},
            }
            for i, c in enumerate(("Lisbon", "Porto"), 1)
        ]
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "message": message,
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


async def _empty_aiter() -> Any:
    return
    yield  # pragma: no cover


def rig_openai(model: str = "gpt-4o") -> tuple[LLM, Callable[[], list[tuple[str, dict]]], Callable[[], Any]]:
    llm = LLM(model, api_key="x")
    client = AsyncMock()

    def _create(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return _empty_aiter()
        return state["completion"]

    state = {"completion": _openai_completion()}
    client.chat.completions.create = AsyncMock(side_effect=_create)
    llm._provider._client = client  # type: ignore[attr-defined]

    def grab() -> list[tuple[str, dict]]:
        return [("chat.completions.create", c.kwargs) for c in client.chat.completions.create.call_args_list]

    def tool_turn() -> Any:
        state["completion"] = _openai_completion(tool_calls=True)
        return client

    return llm, grab, tool_turn


def _meta_response(tool_calls: bool = False) -> Any:
    from openai.types.responses import Response as SDKResponse

    output: list[dict[str, Any]] = [
        {
            "id": "rs_1",
            "type": "reasoning",
            "summary": [],
            "content": None,
            "encrypted_content": "enc-rs_1",
            "status": "completed",
        },
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "phase": "commentary" if tool_calls else None,
            "content": [
                {"type": "output_text", "text": "Let me check.", "annotations": [], "logprobs": []}
            ],
        },
    ]
    if tool_calls:
        output += [
            {
                "id": f"fc_{i}",
                "type": "function_call",
                "call_id": f"call_{i}",
                "name": "get_weather",
                "arguments": json.dumps({"city": c}),
                "status": "completed",
            }
            for i, c in enumerate(("Lisbon", "Porto"), 1)
        ]
    return SDKResponse.model_validate(
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 1789270716.0,
            "model": "muse-spark-1",
            "status": "completed",
            "incomplete_details": None,
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "store": False,
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 1},
                "total_tokens": 15,
            },
        }
    )


def rig_meta(model: str = "muse-spark-1") -> tuple[LLM, Callable[[], list[tuple[str, dict]]], Callable[[], Any]]:
    llm = LLM(model, api_key="x")
    client = AsyncMock()
    state = {"response": _meta_response()}

    def _create(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return _empty_aiter()
        return state["response"]

    client.responses.create = AsyncMock(side_effect=_create)
    llm._provider._client = client  # type: ignore[attr-defined]

    def grab() -> list[tuple[str, dict]]:
        return [("responses.create", c.kwargs) for c in client.responses.create.call_args_list]

    def tool_turn() -> Any:
        state["response"] = _meta_response(tool_calls=True)
        return client

    return llm, grab, tool_turn


def _gemini_response(tool_calls: bool = False) -> Any:
    from google.genai import types

    parts = [types.Part(text="Let me check.")]
    if tool_calls:
        parts += [
            types.Part(
                function_call=types.FunctionCall(id=f"call_{i}", name="get_weather", args={"city": c}),
                thought_signature=b"sig",
            )
            for i, c in enumerate(("Lisbon", "Porto"), 1)
        ]
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=parts),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=5
        ),
    )


def rig_gemini(model: str = "gemini-2.5-flash") -> tuple[LLM, Callable[[], list[tuple[str, dict]]], Callable[[], Any]]:
    llm = LLM(model, api_key="x")
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=_gemini_response())

    async def _stream(**_: Any) -> Any:
        return _empty_aiter()

    client.aio.models.generate_content_stream = AsyncMock(side_effect=_stream)
    llm._provider._client = client  # type: ignore[attr-defined]

    def grab() -> list[tuple[str, dict]]:
        calls = [("aio.models.generate_content", c.kwargs) for c in client.aio.models.generate_content.call_args_list]
        calls += [
            ("aio.models.generate_content_stream", c.kwargs)
            for c in client.aio.models.generate_content_stream.call_args_list
        ]
        return calls

    def tool_turn() -> Any:
        client.aio.models.generate_content.return_value = _gemini_response(tool_calls=True)
        return client

    return llm, grab, tool_turn


def rig_xai(model: str = "grok-4") -> tuple[LLM, Callable[[], list[tuple[str, dict]]], Callable[[], Any]]:
    llm = LLM(model, api_key="x")
    client = MagicMock()
    state = {"tool_calls": []}

    def _chat(**_: Any) -> Any:
        chat = MagicMock()
        chat.sample = AsyncMock(
            return_value=SimpleNamespace(
                content="Let me check.",
                finish_reason="FINISH_REASON_STOP",
                tool_calls=state["tool_calls"],
                reasoning_content="",
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                cost_usd=None,
            )
        )
        chat.stream = _empty_aiter
        return chat

    client.chat.create = MagicMock(side_effect=_chat)
    llm._provider._client = client  # type: ignore[attr-defined]

    def grab() -> list[tuple[str, dict]]:
        return [("chat.create", c.kwargs) for c in client.chat.create.call_args_list]

    def tool_turn() -> Any:
        state["tool_calls"] = [
            SimpleNamespace(
                id=f"call_{i}",
                function=SimpleNamespace(name="get_weather", arguments=json.dumps({"city": c})),
            )
            for i, c in enumerate(("Lisbon", "Porto"), 1)
        ]
        return client

    return llm, grab, tool_turn


RIGS: dict[str, Callable[..., Any]] = {
    "anthropic": rig_anthropic,
    "openai": rig_openai,
    "meta": rig_meta,
    "gemini": rig_gemini,
    "xai": rig_xai,
}


@dataclass(slots=True)
class Captured:
    scenario: str
    method: str
    kwargs: dict[str, Any] | None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


async def _drain(stream: Any) -> None:
    async for _ in stream:
        pass


async def capture(adapter: str, *, model: str | None = None) -> list[Captured]:
    """Run every scenario against one adapter; return the kwargs its mocked client received."""
    out: list[Captured] = []
    for sc in scenarios():
        rig = RIGS[adapter]
        llm, grab, tool_turn = rig(model) if model else rig()
        caught: list[str] = []
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                messages = sc.messages() if callable(sc.messages) else sc.messages
                if messages == "__replay__":
                    # A real first turn through the mocked client, replayed via to_message().
                    tool_turn()
                    first = await llm.complete("Weather in Lisbon and Porto?", **sc.call)
                    messages = [
                        user("Weather in Lisbon and Porto?"),
                        first.to_message(),
                        *[
                            tool_result("sunny", tool_use_id=tc.id, name=tc.name)
                            for tc in first.tool_calls
                        ],
                    ]
                    before = len(grab())
                    await llm.complete(messages, **sc.call)
                    calls = grab()[before:]
                elif sc.mode == "complete":
                    await llm.complete(messages, **sc.call)
                    calls = grab()
                elif sc.mode == "stream":
                    await _drain(llm.stream(messages, **sc.call))
                    calls = grab()
                else:
                    await _drain(llm.stream_events(messages, **sc.call))
                    calls = grab()
                caught = [f"{x.category.__name__}: {x.message}" for x in w]
            for method, kwargs in calls:
                out.append(Captured(sc.name, method, kwargs, warnings=caught))
            if not calls:
                out.append(Captured(sc.name, "-", None, error="no SDK call captured", warnings=caught))
        except Exception as exc:  # adapter refused the scenario (a finding in itself)
            out.append(Captured(sc.name, "-", None, error=f"{type(exc).__name__}: {exc}", warnings=caught))
    return out


def _short(value: Any, limit: int = 2000) -> str:
    text = json.dumps(value, default=lambda o: f"<{type(o).__module__}.{type(o).__name__}>", indent=1)
    return text if len(text) <= limit else text[:limit] + " ..."


if __name__ == "__main__":
    import asyncio
    import sys

    adapter = sys.argv[1]
    for cap in asyncio.run(capture(adapter)):
        print(f"=== {adapter} :: {cap.scenario} :: {cap.method}")
        if cap.error:
            print("   ERROR:", cap.error)
        for wmsg in cap.warnings:
            print("   WARN:", wmsg)
        if cap.kwargs is not None:
            print(_short(cap.kwargs))
