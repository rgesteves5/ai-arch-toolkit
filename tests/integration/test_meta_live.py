"""Meta Model API (Muse Spark) behaviours only the real API can confirm.

Run with ``pytest -m live_api -k meta``. Each test makes a few small calls (fractions of a cent).
Meta's backend often answers 503 ``service_overloaded``; the retry config rides that out.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from ai_arch_toolkit import LLM, RetryConfig, ToolGroup, run_tools, tool, user
from ai_arch_toolkit.toolkit.agents import Agent, ReasoningSpec
from tests.integration.conftest import skip_no_meta

pytestmark = [pytest.mark.integration, pytest.mark.live_api, skip_no_meta]

MODEL = "muse-spark-1.3"
RETRY = RetryConfig(max_retries=5, base_delay=5.0, max_delay=40.0)
WEATHER = {"lisbon": "Sunny, 24C", "porto": "Cloudy, 19C"}


@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city.

    Args:
        city: City name.
    """
    return WEATHER.get(city.lower(), "Unknown")


class Legs(BaseModel):
    animal: str
    legs: int


def _llm() -> LLM:
    return LLM(MODEL, temperature=1.0, max_tokens=4096, retry=RETRY)


def _record_requests(llm: LLM) -> list[dict[str, Any]]:
    """Capture every ``responses.create`` request the provider sends."""
    client = llm._provider._client  # type: ignore[attr-defined]
    original = client.responses.create
    sent: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        sent.append(kwargs)
        return await original(**kwargs)

    client.responses.create = create
    return sent


@pytest.mark.timeout(240)
async def test_completion_is_routed_priced_and_stateless() -> None:
    async with _llm() as llm:
        response = await llm.complete("Reply with exactly the word OK.")

    assert response.text == "OK"
    assert response.stop_reason == "completed"
    assert response.usage.output_tokens > 0
    assert response.cost is not None and response.cost > 0
    assert response.raw.store is False


@pytest.mark.timeout(300)
async def test_tool_loop_replays_the_encrypted_reasoning() -> None:
    group = ToolGroup(get_weather)
    messages = [user("What is the weather in Lisbon? Use the tool, then answer in one sentence.")]
    async with _llm() as llm:
        sent = _record_requests(llm)
        first = await llm.complete(messages, tools=group)
        results = await run_tools(first, group)
        final = await llm.complete([*messages, first.to_message(), *results], tools=group)

    assert [call.name for call in first.tool_calls] == ["get_weather"]
    replayed = [item for item in sent[-1]["input"] if item.get("type") == "reasoning"]
    assert replayed and all(item.get("encrypted_content") for item in replayed)
    assert "24" in final.text


@pytest.mark.timeout(300)
async def test_react_agent_uses_the_tool_results() -> None:
    async with _llm() as llm:
        agent = Agent(ReasoningSpec(strategy="react"), llm, ToolGroup(get_weather))
        result = await agent.run("What is the weather in Lisbon and in Porto? Be brief.")

    assert "24" in result.text and "19" in result.text
    assert result.cost is not None and result.cost > 0


@pytest.mark.timeout(300)
async def test_streamed_tool_turn_replays_into_a_streamed_answer() -> None:
    group = ToolGroup(get_weather)
    prompt = user("What is the weather in Lisbon and in Porto? Call the tool for both cities.")
    async with _llm() as llm:
        sent = _record_requests(llm)
        events = llm.stream_events([prompt], tools=group, thinking=True, thinking_effort="low")
        calls = [event.tool_call async for event in events if event.kind == "tool_call"]
        first = events.response
        assert first is not None
        results = await run_tools(first, group)
        answer = llm.stream([prompt, first.to_message(), *results], tools=group)
        text = "".join([chunk async for chunk in answer])

    assert sorted(call.input["city"].lower() for call in calls) == ["lisbon", "porto"]
    assert first.usage.output_tokens > 0
    replayed = [item for item in sent[-1]["input"] if item.get("type") == "reasoning"]
    assert replayed and all(item.get("encrypted_content") for item in replayed)
    assert "24" in text and "19" in text


@pytest.mark.timeout(240)
async def test_structured_output_and_json_mode() -> None:
    async with _llm() as llm:
        legs = await llm.complete("How many legs does a spider have?", output_schema=Legs)
        city = await llm.complete("Return a JSON object whose key city is Lisbon.", json_mode=True)

    assert isinstance(legs.parsed, Legs)
    assert legs.parsed.legs == 8
    assert '"Lisbon"' in city.text


@pytest.mark.timeout(120)
async def test_count_tokens_with_the_input_tokens_endpoint() -> None:
    async with _llm() as llm:
        count = await llm.count_tokens("What is the weather in Lisbon?", tools=[get_weather])

    assert count > 0
