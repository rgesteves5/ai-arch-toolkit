"""Every tool call gets an id its result can name (G-23), on a loopback OpenAI-compatible server.

A compatible server (Ollama, LM Studio, llama.cpp, vLLM) may send a tool call with no ``id``, an
empty one, or the id of another call. The tools run, and their results must name each call: the
toolkit gives such a call an id of its own before anyone sees it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ai_arch_toolkit import LLM, ToolGroup, run_tools, tool, user
from tests.integration import fakeserver

pytestmark = pytest.mark.integration

PROMPT = user("What is the weather in Porto and in Lisbon?")


@tool
def get_weather(city: str) -> str:
    """The weather in a city."""
    return f"Sunny in {city}"


def _call(city: str, **identity: str) -> dict[str, Any]:
    arguments = json.dumps({"city": city})
    return {"type": "function", "function": {"name": "get_weather", "arguments": arguments}} | (
        identity
    )


def _completion(message: dict[str, Any], finish: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "llama3.2",
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _calls(*calls: dict[str, Any]) -> fakeserver.Reply:
    message = {"role": "assistant", "content": None, "tool_calls": list(calls)}
    return fakeserver.Reply(body=_completion(message, "tool_calls"))


ANSWER = fakeserver.Reply(
    body=_completion({"role": "assistant", "content": "Sunny in both."}, "stop")
)


def _llm(port: int) -> LLM:
    return LLM("llama3.2", base_url=f"http://127.0.0.1:{port}/v1", max_tokens=64)


async def _two_turns(replies: list[fakeserver.Reply]) -> tuple[list[str], dict[str, Any]]:
    """The ids of the first turn's calls, and the second request, which sends their results."""
    group = ToolGroup(get_weather)
    server, port, stats = await fakeserver.start("script", replies=replies)
    async with server, _llm(port) as llm:
        first = await llm.complete([PROMPT], tools=group)
        results = await run_tools(first, group)
        await llm.complete([PROMPT, first.to_message(), *results], tools=group)
    return [call.id for call in first.tool_calls], json.loads(stats.bodies[1])


def _replayed(sent: dict[str, Any]) -> tuple[list[str], list[str]]:
    """The call ids of the replayed assistant turn, and the ids its tool results name."""
    assistant = next(m for m in sent["messages"] if m.get("tool_calls"))
    tools = [m for m in sent["messages"] if m["role"] == "tool"]
    return [c["id"] for c in assistant["tool_calls"]], [m["tool_call_id"] for m in tools]


async def test_calls_without_an_id_get_ids_their_results_name() -> None:
    ids, sent = await _two_turns([_calls(_call("Porto"), _call("Lisbon", id="")), ANSWER])

    assert all(ids)
    assert len(set(ids)) == 2
    assert _replayed(sent) == (ids, ids)


async def test_a_repeated_id_is_replaced_and_the_first_kept() -> None:
    calls = _calls(_call("Porto", id="call_0"), _call("Lisbon", id="call_0"))

    ids, sent = await _two_turns([calls, ANSWER])

    assert ids[0] == "call_0"
    assert ids[1] not in ("", "call_0")
    assert _replayed(sent) == (ids, ids)


def _chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    return fakeserver.sse(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "llama3.2",
            "choices": [choice],
        }
    )


async def test_a_streamed_call_without_an_id_gets_one_in_its_event_and_its_response() -> None:
    call = {"index": 0} | _call("Porto")
    events = [
        _chunk({"role": "assistant", "tool_calls": [call]}),
        _chunk({}, "tool_calls"),
        fakeserver.sse("[DONE]"),
    ]
    server, port, _ = await fakeserver.start("sse", events=events)
    async with server, _llm(port) as llm:
        stream = llm.stream_events([PROMPT], tools=ToolGroup(get_weather))
        seen = [event.tool_call async for event in stream if event.kind == "tool_call"]
        response = stream.response

    assert response is not None
    assert [c.id for c in seen] == [c.id for c in response.tool_calls]
    assert seen[0] is not None
    assert seen[0].id
    assert seen[0].input == {"city": "Porto"}
