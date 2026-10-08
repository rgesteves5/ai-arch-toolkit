"""Tools built at run time from a complete JSON Schema: ``tool_from_schema`` (C02, D62)."""

from __future__ import annotations

import asyncio
import copy
import inspect
import threading
import time
from typing import Any

import pytest

from ai_arch_toolkit.core import MeterScope, ToolGroup, prepare_tools, tool_from_schema
from ai_arch_toolkit.core._content import user
from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider
from ai_arch_toolkit.core._providers._gemini import GeminiProvider
from ai_arch_toolkit.core._providers._openai import OpenAIProvider
from ai_arch_toolkit.core._response import Response, ToolCall
from ai_arch_toolkit.core._tools._approval import ApprovalDecision, ApprovalRequest
from ai_arch_toolkit.core._tools._definition import ToolRuntimePolicy
from ai_arch_toolkit.core._tools._executor import async_execute_tool, execute_tool
from ai_arch_toolkit.core._tools._result import ToolFailure, ToolResult
from ai_arch_toolkit.toolkit import run_tools
from tests.provider_calls import prepare

MODES = ["sync", "async"]

OBJECT: dict[str, Any] = {"type": "object", "properties": {}}

# The shape an MCP server sends: a dialect, titles, a default, a closed root, a oneOf, an x- key
# and a local $defs table (https://modelcontextprotocol.io/specification/2026-07-28/server/tools).
MCP_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "create_issue",
    "type": "object",
    "properties": {
        "title": {"type": "string", "title": "Title"},
        "labels": {"type": "array", "items": {"$ref": "#/$defs/Label"}, "default": []},
        "priority": {"oneOf": [{"type": "integer"}, {"type": "string", "enum": ["low", "high"]}]},
    },
    "required": ["title"],
    "additionalProperties": False,
    "x-mcp-header": "Api-Version",
    "$defs": {"Label": {"type": "string", "title": "Label"}},
}

# What the tool keeps and sends: the same schema, with its local reference inlined.
MCP_SENT: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "create_issue",
    "type": "object",
    "properties": {
        "title": {"type": "string", "title": "Title"},
        "labels": {"type": "array", "items": {"type": "string", "title": "Label"}, "default": []},
        "priority": {"oneOf": [{"type": "integer"}, {"type": "string", "enum": ["low", "high"]}]},
    },
    "required": ["title"],
    "additionalProperties": False,
    "x-mcp-header": "Api-Version",
}

COUNT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"count": {"$ref": "#/$defs/Count"}},
    "required": ["count"],
    "$defs": {"Count": {"type": "integer", "minimum": 1}},
}


class Recorder:
    """A handler that keeps every dict it receives and the thread it ran on."""

    def __init__(self, answer: object = "ok") -> None:
        self.calls: list[dict[str, Any]] = []
        self.threads: list[int] = []
        self._answer = answer

    def __call__(self, arguments: dict[str, Any]) -> object:
        self.calls.append(arguments)
        self.threads.append(threading.get_ident())
        return self._answer


class Approver:
    def __init__(self) -> None:
        self.requests: list[ApprovalRequest] = []

    def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return ApprovalDecision.approve()


def _call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id="tc_1", name=name, input=arguments)


async def _execute(group: ToolGroup, tool_call: ToolCall, mode: str) -> ToolResult:
    if mode == "sync":
        return group.execute(tool_call)
    return await group.async_execute(tool_call)


# --- The handler ----------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
async def test_the_handler_receives_the_arguments_as_one_dict(mode: str) -> None:
    handler = Recorder()
    headers = tool_from_schema(
        handler,
        name="headers",
        input_schema={
            "type": "object",
            "properties": {"x-api-version": {"type": "string"}, "from": {"type": "integer"}},
        },
    )

    result = await _execute(
        ToolGroup(headers), _call("headers", **{"x-api-version": "2", "from": 1}), mode
    )

    assert result.ok, result.to_model_text()
    assert handler.calls == [{"x-api-version": "2", "from": 1}]


async def test_a_sync_handler_runs_in_a_thread_off_the_loop() -> None:
    handler = Recorder()
    slow = tool_from_schema(handler, name="slow", input_schema=OBJECT)

    result = await ToolGroup(slow).async_execute(_call("slow"))

    assert result.ok
    assert handler.threads and handler.threads[0] != threading.get_ident()


async def test_an_async_handler_runs_on_the_loop() -> None:
    loops: list[asyncio.AbstractEventLoop] = []

    async def handler(arguments: dict[str, Any]) -> str:
        loops.append(asyncio.get_running_loop())
        return f"got {arguments}"

    fetch = tool_from_schema(handler, name="fetch", input_schema=OBJECT)
    result = await ToolGroup(fetch).async_execute(_call("fetch"))

    assert result.value == "got {}"
    assert loops == [asyncio.get_running_loop()]


def test_an_async_handler_runs_on_the_sync_path_too() -> None:
    async def handler(arguments: dict[str, Any]) -> str:
        await asyncio.sleep(0)
        return f"n={arguments['n']}"

    count = tool_from_schema(
        handler,
        name="count",
        input_schema={"type": "object", "properties": {"n": {"type": "integer"}}},
    )

    assert ToolGroup(count).execute(_call("count", n="4")).value == "n=4"


@pytest.mark.parametrize("mode", MODES)
async def test_a_tool_result_from_the_handler_passes_intact(mode: str) -> None:
    answer = ToolResult.success({"rows": 2}, metadata={"source": "mcp"})
    table = tool_from_schema(Recorder(answer), name="table", input_schema=OBJECT)

    result = await _execute(ToolGroup(table), _call("table"), mode)

    assert result == answer


@pytest.mark.parametrize("mode", MODES)
async def test_a_tool_failure_keeps_its_type_and_other_exceptions_are_runtime_errors(
    mode: str,
) -> None:
    def missing(arguments: dict[str, Any]) -> str:
        raise ToolFailure("not_found", "no issue 7; list the issues first")

    def broken(arguments: dict[str, Any]) -> str:
        raise RuntimeError("backend down")

    group = ToolGroup(
        tool_from_schema(missing, name="missing", input_schema=OBJECT),
        tool_from_schema(broken, name="broken", input_schema=OBJECT),
    )

    not_found = await _execute(group, _call("missing"), mode)
    runtime = await _execute(group, _call("broken"), mode)

    assert not_found.error is not None and not_found.error.type == "not_found"
    assert runtime.error is not None and runtime.error.type == "runtime_error"
    assert runtime.error.message == "backend down"


def test_the_tool_takes_keywords_and_hides_its_handler() -> None:
    def handler(arguments: dict[str, Any]) -> str:
        return "ok"

    made = tool_from_schema(handler, name="made", description="Make things.", input_schema=OBJECT)

    parameters = list(inspect.signature(made).parameters.values())
    assert [(p.name, p.kind) for p in parameters] == [("arguments", inspect.Parameter.VAR_KEYWORD)]
    assert not hasattr(made, "__wrapped__")
    assert made(a=1) == "ok"
    definition = made.__tool_definition__  # type: ignore[attr-defined]
    assert definition.fn is made
    assert definition.schema.description == "Make things."
    assert definition.policy == ToolRuntimePolicy()


# --- Validation, approval and metering --------------------------------------------


@pytest.mark.parametrize("mode", MODES)
async def test_a_local_reference_is_inlined_and_its_value_coerced_before_approval(
    mode: str,
) -> None:
    handler = Recorder()
    approver = Approver()
    counted = tool_from_schema(
        handler,
        name="counted",
        input_schema=COUNT_SCHEMA,
        policy=ToolRuntimePolicy(requires_approval=True),
    )

    result = await _execute(
        ToolGroup(counted, approval_handler=approver), _call("counted", count="3"), mode
    )

    assert result.ok, result.to_model_text()
    assert approver.requests[0].arguments == {"count": 3}
    assert handler.calls == [{"count": 3}]
    assert counted.__tool_definition__.schema.input_schema == {  # type: ignore[attr-defined]
        "type": "object",
        "properties": {"count": {"type": "integer", "minimum": 1}},
        "required": ["count"],
    }


@pytest.mark.parametrize("mode", MODES)
async def test_without_an_approval_handler_the_call_is_denied_and_the_handler_never_runs(
    mode: str,
) -> None:
    handler = Recorder()
    guarded = tool_from_schema(
        handler,
        name="guarded",
        input_schema=OBJECT,
        policy=ToolRuntimePolicy(capability="github", risk_level="high", requires_approval=True),
    )

    result = await _execute(ToolGroup(guarded), _call("guarded"), mode)

    assert result.error is not None and result.error.type == "approval_denied"
    assert handler.calls == []


async def test_the_meter_counts_the_call_that_ran_and_not_the_one_denied() -> None:
    free = tool_from_schema(Recorder(), name="free", input_schema=OBJECT)
    guarded = tool_from_schema(
        Recorder(),
        name="guarded",
        input_schema=OBJECT,
        policy=ToolRuntimePolicy(requires_approval=True),
    )
    group = ToolGroup(free, guarded)

    with MeterScope() as scope:
        ran = await group.async_execute(_call("free"))
        denied = await group.async_execute(_call("guarded"))

    assert ran.ok and not denied.ok
    assert scope.snapshot().tool_calls == 1


async def test_the_tool_runs_through_every_door() -> None:
    handler = Recorder()
    echo = tool_from_schema(
        handler,
        name="echo",
        input_schema={"type": "object", "properties": {"n": {"type": "integer"}}},
    )
    response = Response(tool_calls=(_call("echo", n="1"),))

    assert prepare_tools([echo]) == [echo.__tool_definition__.schema.to_provider_dict()]  # type: ignore[attr-defined]
    assert execute_tool(_call("echo", n="2"), [echo]).ok
    assert (await async_execute_tool(_call("echo", n="3"), [echo])).ok
    assert [r["content"] for r in await run_tools(response, [echo])] == ["ok"]
    assert handler.calls == [{"n": 2}, {"n": 3}, {"n": 1}]


# --- The schema the tool keeps -----------------------------------------------------


def test_an_mcp_schema_is_kept_as_it_came_with_its_local_references_inlined() -> None:
    created = tool_from_schema(Recorder(), name="create_issue", input_schema=MCP_SCHEMA)

    assert created.__tool_definition__.schema.input_schema == MCP_SENT  # type: ignore[attr-defined]


def test_the_tool_keeps_a_copy_of_the_schema() -> None:
    schema = copy.deepcopy(MCP_SCHEMA)
    created = tool_from_schema(Recorder(), name="create_issue", input_schema=schema)

    schema["properties"]["title"]["type"] = "integer"
    schema["required"].append("labels")

    assert created.__tool_definition__.schema.input_schema == MCP_SENT  # type: ignore[attr-defined]


def test_a_recursive_reference_stays_a_reference_with_its_table() -> None:
    tree = {
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}},
            }
        },
    }

    kept = tool_from_schema(Recorder(), name="tree", input_schema=tree)

    schema = kept.__tool_definition__.schema.input_schema  # type: ignore[attr-defined]
    assert schema["properties"]["root"]["properties"]["children"]["items"] == {
        "$ref": "#/$defs/Node"
    }
    assert schema["$defs"] == tree["$defs"]


@pytest.mark.parametrize(
    ("schema", "words"),
    [
        ({"type": "array", "items": {}}, '"type": "object"'),
        ({"properties": {"q": {"type": "string"}}}, '"type": "object"'),
        (
            {"type": "object", "properties": {"q": {"$ref": "https://example.com/q.json"}}},
            "https://example.com/q.json",
        ),
        ({"type": "object", "properties": {"q": {"$ref": "other.json#/q"}}}, "other.json#/q"),
        ({"type": "object", "properties": {"x": {"maximum": float("nan")}}}, "JSON"),
        ({"type": "object", "properties": {"x": {"enum": {1, 2}}}}, "JSON"),
        (
            {
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/Any"}},
                "$defs": {"Any": True},
            },
            "'#/$defs/Any'",
        ),
        (
            {"type": "object", "properties": {"x": {"$ref": "#/$defs/L"}}, "$defs": {"L": [1]}},
            "'#/$defs/L'",
        ),
        ({"$ref": "#/$defs/Root", "$defs": {"Root": False}}, "'#/$defs/Root'"),
    ],
    ids=[
        "array root",
        "no type",
        "remote ref",
        "relative file ref",
        "NaN",
        "a set",
        "a boolean definition",
        "a list definition",
        "a root that refers to a boolean",
    ],
)
def test_a_schema_the_tool_cannot_send_is_a_value_error(
    schema: dict[str, Any], words: str
) -> None:
    with pytest.raises(ValueError, match="input_schema") as raised:
        tool_from_schema(Recorder(), name="bad", input_schema=schema)

    assert words in str(raised.value)


@pytest.mark.parametrize("name", ["a.b", "1a", "a" * 65, "", "get weather", "café"])
def test_a_name_outside_the_portable_rule_is_a_value_error(name: str) -> None:
    with pytest.raises(ValueError, match="portable"):
        tool_from_schema(Recorder(), name=name, input_schema=OBJECT)


@pytest.mark.parametrize("name", ["a", "_private", "github-search", "A" * 64, "x_1-2"])
def test_a_portable_name_is_accepted(name: str) -> None:
    made = tool_from_schema(Recorder(), name=name, input_schema=OBJECT)

    assert made.__tool_definition__.schema.name == name  # type: ignore[attr-defined]


def _doubling_bomb(levels: int) -> dict[str, Any]:
    """Each definition refers twice to the next one: ``2**levels`` copies once inlined."""
    definitions: dict[str, Any] = {
        f"D{i}": {
            "type": "object",
            "properties": {"a": {"$ref": f"#/$defs/D{i + 1}"}, "b": {"$ref": f"#/$defs/D{i + 1}"}},
        }
        for i in range(levels)
    }
    definitions[f"D{levels}"] = {"type": "string"}
    return {"type": "object", "properties": {"root": {"$ref": "#/$defs/D0"}}, "$defs": definitions}


def _wide_bomb(copies: int) -> dict[str, Any]:
    """One long definition referred to many times: small to send, large once inlined."""
    properties = {f"p{i}": {"$ref": "#/$defs/Long"} for i in range(copies)}
    long = {"type": "string", "description": "x" * 100_000}
    return {"type": "object", "properties": properties, "$defs": {"Long": long}}


def _chain(length: int) -> dict[str, Any]:
    """Each definition holds the next one: a schema that nests ``length`` levels once inlined."""
    definitions: dict[str, Any] = {
        f"D{i}": {"type": "object", "properties": {"next": {"$ref": f"#/$defs/D{i + 1}"}}}
        for i in range(length)
    }
    definitions[f"D{length}"] = {"type": "string"}
    return {"type": "object", "properties": {"root": {"$ref": "#/$defs/D0"}}, "$defs": definitions}


def _ref_chain(length: int) -> dict[str, Any]:
    """Each definition is only a reference to the next one: no nesting until the last."""
    definitions: dict[str, Any] = {f"D{i}": {"$ref": f"#/$defs/D{i + 1}"} for i in range(length)}
    definitions[f"D{length}"] = {"type": "string"}
    return {"type": "object", "properties": {"root": {"$ref": "#/$defs/D0"}}, "$defs": definitions}


def _nested_any_of(levels: int) -> dict[str, Any]:
    """A property nested ``levels`` deep in ``anyOf`` branches, with no references at all."""
    node: dict[str, Any] = {"type": "integer"}
    for _ in range(levels):
        node = {"anyOf": [node, {"type": "null"}]}
    return {"type": "object", "properties": {"n": node}}


def _doubling_any_of(levels: int) -> dict[str, Any]:
    """Each definition offers the next one twice, as ``anyOf`` branches: ``2**levels`` leaves."""
    definitions: dict[str, Any] = {
        f"D{i}": {"anyOf": [{"$ref": f"#/$defs/D{i + 1}"}, {"$ref": f"#/$defs/D{i + 1}"}]}
        for i in range(levels)
    }
    definitions[f"D{levels}"] = {"type": "integer"}
    return {"type": "object", "properties": {"x": {"$ref": "#/$defs/D0"}}, "$defs": definitions}


@pytest.mark.parametrize(
    "schema",
    [
        _doubling_bomb(30),
        _wide_bomb(50),
        _chain(5_000),
        _ref_chain(2_000),
        _nested_any_of(300),
        _doubling_any_of(12),
    ],
    ids=[
        "doubling refs",
        "a long definition many times",
        "a long chain",
        "a chain of references alone",
        "deep anyOf",
        "anyOf branches that double",
    ],
)
def test_a_hostile_schema_fails_in_under_a_second(schema: dict[str, Any]) -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="input_schema"):
        tool_from_schema(Recorder(), name="bomb", input_schema=schema)

    assert time.perf_counter() - started < 1.0


def _deepest_doubling_the_budget_takes() -> int:
    levels = 1
    while True:
        try:
            tool_from_schema(Recorder(), name="wide", input_schema=_doubling_any_of(levels + 1))
        except ValueError:
            return levels
        levels += 1


def test_a_bad_call_against_many_alternatives_is_refused_quickly_and_briefly() -> None:
    wide = tool_from_schema(Recorder(), name="wide", input_schema=_doubling_any_of(9))  # 512

    started = time.perf_counter()
    refused = ToolGroup(wide).execute(_call("wide", x="abc"))

    assert time.perf_counter() - started < 0.1
    assert refused.error is not None and refused.error.type == "validation_error"
    assert refused.error.message == "Tool 'wide' argument 'x': expected integer, got str 'abc'"


def test_past_the_alternatives_the_walk_takes_a_value_passes_as_it_came() -> None:
    handler = Recorder()
    levels = _deepest_doubling_the_budget_takes()
    wide = tool_from_schema(handler, name="wide", input_schema=_doubling_any_of(levels))

    started = time.perf_counter()
    result = ToolGroup(wide).execute(_call("wide", x="abc"))

    assert time.perf_counter() - started < 0.1
    assert 2**levels > 1_000 and result.ok
    assert handler.calls == [{"x": "abc"}]  # for the handler, or its server, to check


# --- The schema reaches the providers as the tool keeps it -------------------------


def _sent(created: Any) -> dict[str, Any]:
    return created.__tool_definition__.schema.input_schema


def test_the_schema_reaches_anthropic_unchanged() -> None:
    created = tool_from_schema(Recorder(), name="create_issue", input_schema=MCP_SCHEMA)

    params = prepare(
        AnthropicProvider("claude-sonnet-4-6", "test-key"),
        [user("hi")],
        tools=prepare_tools([created]),
        max_tokens=100,
    ).params

    assert params["tools"][0]["input_schema"] == _sent(created) == MCP_SENT


def test_the_schema_reaches_openai_unchanged() -> None:
    created = tool_from_schema(Recorder(), name="create_issue", input_schema=MCP_SCHEMA)

    params = prepare(
        OpenAIProvider("gpt-6-luna", "test-key"), [user("hi")], tools=prepare_tools([created])
    ).params

    assert params["tools"][0]["parameters"] == _sent(created) == MCP_SENT


def test_the_schema_reaches_gemini_unchanged() -> None:
    created = tool_from_schema(Recorder(), name="create_issue", input_schema=MCP_SCHEMA)

    config = prepare(
        GeminiProvider("gemini-3.8-flash", "test-key"),
        [user("hi")],
        tools=prepare_tools([created]),
    ).params["config"]

    declaration = config.tools[0].function_declarations[0]
    assert declaration.parameters_json_schema == _sent(created) == MCP_SENT
