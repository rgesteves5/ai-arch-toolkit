"""Provider behaviours only a real API can confirm. Run with ``pytest -m live_api``.

Each test makes one or two small calls to a cheap model.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from pydantic import BaseModel

from ai_arch_toolkit import LLM, ToolGroup, cache, run_tools, system, tool, user
from tests.integration.conftest import skip_no_anthropic, skip_no_gemini, skip_no_openai

pytestmark = [pytest.mark.integration, pytest.mark.live_api]

ANTHROPIC = "claude-haiku-4-5"
OPENAI = "gpt-4.1-nano"
GEMINI = "gemini-2.5-flash"


@tool
def locate(point: tuple[float, float]) -> str:
    """Describe a coordinate pair.

    Args:
        point: Latitude and longitude.
    """
    return f"lat={point[0]} lon={point[1]}"


@tool
def add(a: int, b: int) -> str:
    """Add two integers.

    Args:
        a: First number.
        b: Second number.
    """
    return str(a + b)


class Address(BaseModel):
    city: str
    country: str = "Portugal"


class Person(BaseModel):
    name: str
    age: int
    nickname: str | None = None
    address: Address


@tool
def inspect_value(value: Any) -> str:
    """Report the Python type of a JSON value.

    Args:
        value: Any JSON value (object, list, number or text).
    """
    return type(value).__name__


ROOT_DEFS_TOOL = {
    "name": "save_item",
    "description": "Save an item.",
    "input_schema": {
        "type": "object",
        "properties": {"item": {"$ref": "#/$defs/Item"}},
        "required": ["item"],
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "qty": {"type": "integer"}},
                "required": ["name", "qty"],
            }
        },
    },
}

MERGE_MESSAGES = [
    system("Always end your answer with the word BANANA."),
    user("Name one yellow fruit."),
]
MERGE_SYSTEM = "ANSWER IN UPPERCASE LETTERS ONLY."


async def _assert_system_prompts_merge(model: str) -> None:
    async with LLM(model) as llm:
        response = await llm.complete(MERGE_MESSAGES, system=MERGE_SYSTEM, max_tokens=60)
    text = response.text.strip()
    assert "BANANA" in text.upper(), text
    assert text.upper() == text, text


@skip_no_anthropic
@pytest.mark.timeout(60)
async def test_anthropic_merges_system_argument_and_system_message() -> None:
    await _assert_system_prompts_merge(ANTHROPIC)


@skip_no_openai
@pytest.mark.timeout(60)
async def test_openai_merges_system_argument_and_system_message() -> None:
    await _assert_system_prompts_merge(OPENAI)


@skip_no_gemini
@pytest.mark.timeout(90)
async def test_gemini_merges_system_argument_and_system_message() -> None:
    await _assert_system_prompts_merge(GEMINI)


@skip_no_anthropic
@pytest.mark.timeout(60)
async def test_anthropic_stream_usage_matches_complete() -> None:
    # message_delta usage is cumulative; adding it to message_start doubled the input tokens.
    messages = [user("Name three colours, comma separated, nothing else.")]
    async with LLM(ANTHROPIC) as llm:
        complete = await llm.complete(messages, max_tokens=40)
        stream = llm.stream(messages, max_tokens=40)
        async for _ in stream:
            pass
        events = llm.stream_events(messages, max_tokens=40)
        async for _ in events:
            pass

    assert stream.response.usage.input_tokens == complete.usage.input_tokens
    assert events.response.usage.input_tokens == complete.usage.input_tokens
    assert stream.response.usage.output_tokens > 0


@skip_no_anthropic
@pytest.mark.timeout(120)
async def test_anthropic_caches_a_system_message_cache_part() -> None:
    # Haiku 4.5 caches prompts of 4096 tokens or more; ~200 records are about 5 600 tokens.
    records = "\n".join(
        f"Record {i}: item number {i} weighs {i * 3} grams, costs {i * 7} cents and is "
        f"{'red' if i % 2 else 'blue'}."
        for i in range(1, 200)
    )
    facts = f"Run {time.time()}\n{records}"  # a fresh prefix, so the first call writes the cache
    cached_system = {"role": "system", "content": ["You answer from the records.", cache(facts)]}
    async with LLM(ANTHROPIC) as llm:
        first = await llm.complete(
            [cached_system, user("What colour is item 41? One word.")], max_tokens=10
        )
        second = await llm.complete(
            [cached_system, user("What colour is item 42? One word.")], max_tokens=10
        )
        counted = await llm.count_tokens([cached_system, user("hi")])

    assert first.usage.cache_write_tokens > 4000, first.usage
    assert second.usage.cache_read_tokens == first.usage.cache_write_tokens, second.usage
    assert counted > 4000


@skip_no_gemini
@pytest.mark.timeout(120)
async def test_gemini_accepts_tuple_and_reference_schemas_as_json_schema() -> None:
    # Both schemas fail the SDK's OpenAPI validation and go through parameters_json_schema.
    async with LLM(GEMINI) as llm:
        located = await llm.complete(
            "Call locate with point [38.7, -9.1]. Then stop.", tools=[locate], max_tokens=300
        )
        saved = await llm.complete(
            "Call save_item with an item named bolt, quantity 4. Then stop.",
            tools=[ROOT_DEFS_TOOL],
            max_tokens=300,
        )

    results = await run_tools(located, ToolGroup(locate))
    assert [r["content"] for r in results] == ["lat=38.7 lon=-9.1"], located.tool_calls
    assert [(c.name, c.input) for c in saved.tool_calls] == [
        ("save_item", {"item": {"name": "bolt", "qty": 4}})
    ]


@skip_no_gemini
@pytest.mark.timeout(120)
async def test_gemini_accepts_openapi_and_json_schema_declarations_together() -> None:
    # `add` fits Gemini's OpenAPI subset (parameters); `locate` does not (parameters_json_schema).
    async with LLM(GEMINI) as llm:
        response = await llm.complete(
            "Call locate with point [38.7, -9.1] and add with a=2, b=3. Call both tools.",
            tools=[add, locate],
            max_tokens=400,
        )

    assert sorted(call.name for call in response.tool_calls) == ["add", "locate"]


@skip_no_openai
@pytest.mark.timeout(60)
async def test_openai_structured_output_accepts_a_plain_pydantic_model() -> None:
    # model_json_schema() has no additionalProperties: false; strict mode used to answer 400.
    async with LLM(OPENAI) as llm:
        response = await llm.complete("Ana, 31, lives in Lisbon.", output_schema=Person)

    assert isinstance(response.parsed, Person), response.text
    assert response.parsed.address.city == "Lisbon"


@skip_no_openai
@pytest.mark.timeout(60)
async def test_untyped_and_tuple_parameters_round_trip_through_run_tools() -> None:
    group = ToolGroup(locate, inspect_value)
    prompt = (
        "Call locate with point [38.7, -9.1]. Also call inspect_value with the JSON object "
        '{"a": [1, 2]}. Call both tools, then stop.'
    )
    async with LLM(OPENAI) as llm:
        response = await llm.complete([user(prompt)], tools=group, max_tokens=300)

    results = await run_tools(response, group)
    by_name = {r["name"]: r["content"] for r in results}
    assert by_name == {"locate": "lat=38.7 lon=-9.1", "inspect_value": "dict"}, response.tool_calls
