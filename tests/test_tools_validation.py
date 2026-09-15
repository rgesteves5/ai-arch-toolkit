"""Tool-call arguments are validated and coerced against the schema before any gate runs."""

from __future__ import annotations

import enum
from typing import Any, Literal

import pytest

from ai_arch_toolkit.core._metering._scope import MeterScope
from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._approval import ApprovalDecision, ApprovalRequest
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._governance import ExecutionContext, GateModify, GateResult
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.core._tools._result import ToolResult

type Count = int


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def halve(x: float) -> float:
    """Halve a number."""
    return x / 2


@tool
def toggle(on: bool) -> str:
    """Switch something on or off."""
    return "on" if on else "off"


@tool
def pick(mode: Literal["fast", "slow"]) -> str:
    """Pick a mode."""
    return mode


@tool
def either(value: int | str) -> str:
    """Accept an int or a string."""
    return f"{type(value).__name__}:{value}"


@tool
def describe(text: str) -> str:
    """Describe the argument's Python type."""
    return type(text).__name__


@tool
def flexible(a: int, **extra: str) -> str:
    """Accept extra keyword arguments."""
    return f"{a!r} {sorted(extra.items())}"


@tool
def repeat(n: Count) -> int:
    """Double a count declared through a type alias."""
    return n * 2


@tool
def buggy(x: int) -> str:
    """A tool whose own body raises TypeError."""
    return "value: " + x  # type: ignore[operator]


@tool
def strict_flag(flag: Literal[True]) -> str:
    """Only accept True."""
    return f"flag={flag!r}"


@tool
def either_flag(flag: Literal[True, False]) -> str:
    """Accept a boolean literal."""
    return f"flag={flag!r}"


class Switch(enum.Enum):
    ON = True
    OFF = False


@tool
def set_switch(state: Switch) -> str:
    """Set a switch from its boolean value."""
    return f"state={state!r}"


@tool(schema={"limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]}})
def paged(limit: int | None = None) -> str:
    """Page through results."""
    return f"limit={limit!r}"


@tool
def find(query: str | None) -> str:
    """Find something; the query may be null."""
    return f"query={query!r}"


@tool(requires_approval=True)
def deploy(target: str, replicas: int) -> str:
    """Deploy a service."""
    return f"{target} x{replicas}"


def _call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id="tc_1", name=name, input=arguments)


async def _execute(group: ToolGroup, tool_call: ToolCall, mode: str) -> ToolResult:
    if mode == "sync":
        return group.execute(tool_call)
    return await group.async_execute(tool_call)


MODES = ["sync", "async"]


@pytest.mark.parametrize("mode", MODES)
async def test_numeric_strings_are_coerced_to_integers(mode: str) -> None:
    result = await _execute(ToolGroup(add), _call("add", a="1", b="2"), mode)

    assert result.ok and result.value == 3


@pytest.mark.parametrize("mode", MODES)
async def test_integral_floats_are_coerced_to_integers(mode: str) -> None:
    result = await _execute(ToolGroup(add), _call("add", a=1.0, b="2.0"), mode)

    assert result.ok and result.value == 3


@pytest.mark.parametrize("mode", MODES)
async def test_a_value_that_does_not_coerce_is_rejected_naming_the_argument(mode: str) -> None:
    result = await _execute(ToolGroup(add), _call("add", a="x", b=1), mode)

    assert not result.ok and result.error is not None
    assert result.error.type == "validation_error"
    assert "'a'" in result.error.message and "integer" in result.error.message
    assert result.error.details["argument"] == "a"


@pytest.mark.parametrize("mode", MODES)
async def test_booleans_are_not_integers(mode: str) -> None:
    result = await _execute(ToolGroup(add), _call("add", a=True, b=1), mode)

    assert result.error is not None and result.error.type == "validation_error"


async def test_numeric_strings_are_coerced_to_numbers() -> None:
    result = ToolGroup(halve).execute(_call("halve", x="3"))

    assert result.ok and result.value == 1.5


async def test_boolean_strings_are_coerced_and_other_strings_rejected() -> None:
    group = ToolGroup(toggle)

    assert group.execute(_call("toggle", on="true")).value == "on"
    assert group.execute(_call("toggle", on="False")).value == "off"
    rejected = group.execute(_call("toggle", on="yes"))
    assert rejected.error is not None and rejected.error.type == "validation_error"


async def test_enum_membership_is_checked() -> None:
    result = ToolGroup(pick).execute(_call("pick", mode="medium"))

    assert result.error is not None and result.error.type == "validation_error"
    assert "fast" in result.error.message


async def test_any_of_keeps_a_value_that_already_matches_a_branch() -> None:
    group = ToolGroup(either)

    assert group.execute(_call("either", value="1")).value == "str:1"
    assert group.execute(_call("either", value=1)).value == "int:1"
    assert group.execute(_call("either", value=1.0)).value == "int:1"


async def test_string_parameters_are_not_coerced() -> None:
    result = ToolGroup(describe).execute(_call("describe", text=123))

    assert result.ok and result.value == "int"


@pytest.mark.parametrize("mode", MODES)
async def test_a_missing_required_argument_is_rejected(mode: str) -> None:
    result = await _execute(ToolGroup(add), _call("add", a=1), mode)

    assert result.error is not None and result.error.type == "validation_error"
    assert "'b'" in result.error.message


@pytest.mark.parametrize("mode", MODES)
async def test_an_unknown_argument_is_rejected_listing_the_expected_ones(mode: str) -> None:
    result = await _execute(ToolGroup(add), _call("add", a=1, b=2, c=3), mode)

    assert result.error is not None and result.error.type == "validation_error"
    assert "'c'" in result.error.message and "a, b" in result.error.message


async def test_a_tool_with_kwargs_accepts_extra_arguments() -> None:
    result = ToolGroup(flexible).execute(_call("flexible", a="1", colour="red"))

    assert result.ok and result.value == "1 [('colour', 'red')]"


async def test_a_type_alias_is_described_and_coerced_as_its_value() -> None:
    schema = repeat.__tool_definition__.schema.input_schema  # type: ignore[attr-defined]

    assert schema["properties"]["n"] == {"type": "integer"}
    assert ToolGroup(repeat).execute(_call("repeat", n="5")).value == 10


@pytest.mark.parametrize("mode", MODES)
async def test_a_type_error_raised_by_the_tool_itself_is_a_runtime_error(mode: str) -> None:
    result = await _execute(ToolGroup(buggy), _call("buggy", x=1), mode)

    assert result.error is not None
    assert result.error.type == "runtime_error"
    assert result.error.retryable


class _Handler:
    def __init__(self, decision: ApprovalDecision | None = None) -> None:
        self.requests: list[ApprovalRequest] = []
        self._decision = decision or ApprovalDecision.approve()

    def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return self._decision


@pytest.mark.parametrize("mode", MODES)
async def test_the_approval_handler_sees_the_coerced_arguments(mode: str) -> None:
    handler = _Handler()
    group = ToolGroup(deploy, approval_handler=handler)

    result = await _execute(group, _call("deploy", target="prod", replicas="3"), mode)

    assert result.ok and result.value == "prod x3"
    assert handler.requests[0].arguments == {"target": "prod", "replicas": 3}


@pytest.mark.parametrize("mode", MODES)
async def test_an_invalid_call_never_reaches_the_approval_handler(mode: str) -> None:
    handler = _Handler()
    group = ToolGroup(deploy, approval_handler=handler)

    result = await _execute(group, _call("deploy", target="prod", replicas="many"), mode)

    assert result.error is not None and result.error.type == "validation_error"
    assert handler.requests == []


async def test_invalid_modified_arguments_from_the_handler_are_rejected() -> None:
    handler = _Handler(
        ApprovalDecision.approve(modified_args={"target": "prod", "replicas": "lots"})
    )
    group = ToolGroup(deploy, approval_handler=handler)

    result = group.execute(_call("deploy", target="prod", replicas=1))

    assert result.error is not None and result.error.type == "validation_error"


class _ForceStaging:
    def check_sync(self, ctx: ExecutionContext) -> GateResult | None:
        return GateModify(args={**ctx.tool_call.input, "target": "staging"})

    async def check(self, ctx: ExecutionContext) -> GateResult | None:
        return self.check_sync(ctx)


@pytest.mark.parametrize("mode", MODES)
async def test_gate_modifications_chain_into_the_approval_request(mode: str) -> None:
    handler = _Handler()
    group = ToolGroup(deploy, gates=[_ForceStaging()], approval_handler=handler)

    result = await _execute(group, _call("deploy", target="prod", replicas=2), mode)

    assert result.ok and result.value == "staging x2"
    assert handler.requests[0].arguments == {"target": "staging", "replicas": 2}


async def test_a_failed_validation_does_not_consume_max_calls() -> None:
    group = ToolGroup(add, max_calls=1)

    rejected = group.execute(_call("add", a="x", b=1))
    accepted = group.execute(_call("add", a=1, b=2))

    assert rejected.error is not None and rejected.error.type == "validation_error"
    assert accepted.ok and accepted.value == 3


async def test_a_failed_validation_is_not_metered() -> None:
    with MeterScope() as scope:
        await ToolGroup(add).async_execute(_call("add", a="x", b=1))

    assert scope.snapshot().tool_calls == 0


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("argument", ["a", "b"])
async def test_an_integer_string_too_long_to_convert_is_a_validation_error(
    mode: str, argument: str
) -> None:
    arguments = {"a": 1, "b": 2, argument: "9" * 5_000}

    result = await _execute(ToolGroup(add), _call("add", **arguments), mode)

    assert result.error is not None and result.error.type == "validation_error"
    assert result.error.details["argument"] == argument


async def test_a_number_string_too_long_to_convert_is_a_validation_error() -> None:
    result = ToolGroup(halve).execute(_call("halve", x="9" * 5_000))

    assert result.error is not None and result.error.type == "validation_error"


async def test_boolean_literals_and_enums_accept_booleans() -> None:
    assert ToolGroup(strict_flag).execute(_call("strict_flag", flag=True)).value == "flag=True"
    assert ToolGroup(either_flag).execute(_call("either_flag", flag="false")).value == "flag=False"
    refused = ToolGroup(strict_flag).execute(_call("strict_flag", flag=1))
    assert refused.error is not None and refused.error.type == "validation_error"
    assert ToolGroup(set_switch).execute(_call("set_switch", state=True)).ok


async def test_a_null_branch_accepts_only_null() -> None:
    group = ToolGroup(paged)

    assert group.execute(_call("paged", limit=None)).value == "limit=None"
    assert group.execute(_call("paged", limit="3")).value == "limit=3"
    refused = group.execute(_call("paged", limit="lots"))
    assert refused.error is not None and refused.error.type == "validation_error"


class _MalformedModify:
    def check_sync(self, ctx: ExecutionContext) -> GateResult | None:
        return GateModify(args=[("a", 1), ("b", 2)])  # type: ignore[arg-type]

    async def check(self, ctx: ExecutionContext) -> GateResult | None:
        return self.check_sync(ctx)


@pytest.mark.parametrize("mode", MODES)
async def test_a_gate_returning_non_mapping_arguments_is_a_validation_error(mode: str) -> None:
    group = ToolGroup(add, gates=[_MalformedModify()])

    result = await _execute(group, _call("add", a=1, b=2), mode)

    assert result.error is not None and result.error.type == "validation_error"


@pytest.mark.parametrize("mode", MODES)
async def test_an_omitted_optional_argument_without_a_default_is_passed_as_none(mode: str) -> None:
    # The schema lets the model omit `query: str | None`; Python still needs the argument.
    result = await _execute(ToolGroup(find), _call("find"), mode)

    assert result.ok, result.to_model_text()
    assert result.value == "query=None"
