"""Provider behaviours only a real API can confirm. Run with ``pytest -m live_api``.

Each test makes one or two small calls to a cheap model.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from pydantic import BaseModel

from ai_arch_toolkit import LLM, ToolGroup, cache, run_tools, system, tool, tool_from_schema, user
from tests.integration.conftest import (
    skip_no_anthropic,
    skip_no_gemini,
    skip_no_meta,
    skip_no_openai,
    skip_no_xai,
)

pytestmark = [pytest.mark.integration, pytest.mark.live_api]

ANTHROPIC = "claude-haiku-4-5"
OPENAI = "gpt-4.1-mini"
GEMINI = "gemini-2.5-flash"
XAI = "grok-4.3"  # the cheapest Grok (test_provider_hardening_live.py)
META = "muse-spark-1.3"  # Meta tunes it for temperature=1.0 (https://dev.meta.ai/docs/reasoning)


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


# --- C02f (D62, C02.7): MCP-shaped schemas from tool_from_schema, as each provider gets them ---
# The tool sends the schema as it came, its local references inlined. A provider that refuses one
# of these gets a rule in its adapter, with its source, instead of a schema cleaned for everyone.

MCP_ISSUE: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "create_issue",
    "type": "object",
    "properties": {
        "title": {"type": "string", "title": "Title", "description": "The issue's title."},
        "priority": {
            "title": "Priority",
            "description": "From 1 (urgent) to 5, or the words low or high.",
            "oneOf": [
                {"type": "integer", "minimum": 1, "maximum": 5},
                {"type": "string", "enum": ["low", "high"]},
            ],
        },
        "labels": {"type": "array", "items": {"$ref": "#/$defs/Label"}, "default": []},
    },
    "required": ["title", "priority"],
    "additionalProperties": False,
    "x-mcp-header": "Api-Version",
    "$defs": {"Label": {"type": "string", "title": "Label", "enum": ["bug", "docs"]}},
}

# A recursive reference stays one: the provider receives the $defs table at the root.
MCP_TREE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "The folder to list."},
        "filter": {"$ref": "#/$defs/Filter", "description": "Optional filter tree."},
    },
    "required": ["path"],
    "$defs": {
        "Filter": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "any": {"type": "array", "items": {"$ref": "#/$defs/Filter"}},
            },
        }
    },
}

# Keys that are not Python names, as MCP servers send them (the reason the handler takes a dict).
# The schema fits Gemini's OpenAPI subset, so it goes in `parameters`, whose names google-genai
# documents as [A-Za-z_][A-Za-z0-9_]{0,63} (FunctionDeclaration): "x-request-id" breaks that rule.
MCP_HEADERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "from": {"type": "string", "description": "Who sends the message."},
        "x-request-id": {"type": "string", "description": "The request id to echo."},
    },
    "required": ["from", "x-request-id"],
    "additionalProperties": False,
}

TOOL_MODELS = [
    pytest.param(ANTHROPIC, {}, marks=skip_no_anthropic, id="anthropic"),
    pytest.param(OPENAI, {}, marks=skip_no_openai, id="openai"),
    pytest.param(GEMINI, {}, marks=skip_no_gemini, id="gemini"),
    pytest.param(XAI, {}, marks=skip_no_xai, id="xai"),
    pytest.param(META, {"temperature": 1.0}, marks=skip_no_meta, id="meta"),
]


async def _call_once(
    model: str, options: dict[str, Any], prompt: str, schema: dict[str, Any], name: str
) -> list[dict[str, Any]]:
    """Ask ``model`` to call a ``tool_from_schema`` tool, run the call, and return what it got."""
    received: list[dict[str, Any]] = []

    def handler(arguments: dict[str, Any]) -> str:
        received.append(arguments)
        return "done"

    group = ToolGroup(
        tool_from_schema(handler, name=name, description=prompt, input_schema=schema)
    )
    async with LLM(model) as llm:
        response = await llm.complete(
            f"{prompt} Call the {name} tool once, then stop.",
            tools=group,
            max_tokens=1024,
            **options,
        )
    results = await run_tools(response, group)

    assert [call.name for call in response.tool_calls] == [name], response.text
    assert [result["content"] for result in results] == ["done"], results
    return received


@pytest.mark.parametrize(("model", "options"), TOOL_MODELS)
@pytest.mark.timeout(120)
async def test_an_mcp_shaped_schema_is_accepted_and_called(
    model: str, options: dict[str, Any]
) -> None:
    received = await _call_once(
        model,
        options,
        "Create an issue titled 'Login fails' with priority 2 and the label bug.",
        MCP_ISSUE,
        "create_issue",
    )

    assert received[0]["title"] == "Login fails", received
    assert received[0]["priority"] == 2, received
    assert received[0].get("labels") == ["bug"], received


@pytest.mark.parametrize(("model", "options"), TOOL_MODELS)
@pytest.mark.timeout(120)
async def test_a_recursive_reference_with_its_table_is_accepted_and_called(
    model: str, options: dict[str, Any]
) -> None:
    received = await _call_once(
        model, options, "List the folder /docs, with no filter.", MCP_TREE, "list_folder"
    )

    assert received[0]["path"] == "/docs", received


@pytest.mark.parametrize(("model", "options"), TOOL_MODELS)
@pytest.mark.timeout(120)
async def test_argument_names_that_are_not_python_names_are_accepted_and_called(
    model: str, options: dict[str, Any]
) -> None:
    received = await _call_once(
        model,
        options,
        "Send a message from alice with the request id r-1.",
        MCP_HEADERS,
        "send_message",
    )

    assert received == [{"from": "alice", "x-request-id": "r-1"}], received
