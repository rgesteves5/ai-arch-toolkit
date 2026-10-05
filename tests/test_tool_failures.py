"""A tool that cannot answer raises ``ToolFailure``; the executor returns it as a typed failure,
and each provider receives the result marked as an error (T01, D37, D42)."""

from __future__ import annotations

import copy
import pickle

import pytest

from ai_arch_toolkit.core import (
    MeterScope,
    Response,
    RunConfig,
    ToolCall,
    ToolFailure,
    ToolGroup,
    tool,
    tool_result,
)
from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider
from ai_arch_toolkit.core._providers._gemini import GeminiProvider
from ai_arch_toolkit.core._tools._executor import async_execute_tool, execute_tool
from ai_arch_toolkit.toolkit import run_tools, run_tools_sync
from tests.provider_calls import prepare


@tool
def find_page(title: str) -> str:
    """Find a page by its title."""
    if title == "missing":
        raise ToolFailure("not_found", f"no page {title!r}; search with page_search")
    if title == "secret":
        raise ToolFailure("upstream", "the wiki said: bad key sk-ant-api03-abcdefghijklmnop")
    if title == "busy":
        raise ToolFailure("rate_limited", "slow down", retryable=True, details={"status": 429})
    return f"the page {title!r}"


@tool
def broken(title: str) -> str:
    """A tool with a bug."""
    raise KeyError(title)


def _call(name: str, title: str) -> ToolCall:
    return ToolCall(id="t1", name=name, input={"title": title})


def test_a_tool_failure_is_a_typed_failed_result() -> None:
    result = execute_tool(_call("find_page", "missing"), [find_page])

    assert not result.ok and result.error is not None
    assert result.error.type == "not_found"
    assert result.error.message == "no page 'missing'; search with page_search"
    assert result.error.retryable is False
    assert result.error.details == {"tool_name": "find_page"}
    assert result.to_model_text() == (
        "Tool error [not_found]: no page 'missing'; search with page_search"
    )


async def test_the_async_path_keeps_the_type_and_retryable() -> None:
    result = await async_execute_tool(_call("find_page", "busy"), [find_page])

    assert result.error is not None
    assert (result.error.type, result.error.retryable) == ("rate_limited", True)
    assert result.error.details == {"status": 429, "tool_name": "find_page"}


def test_a_failures_message_is_redacted() -> None:
    result = execute_tool(_call("find_page", "secret"), [find_page])

    assert result.error is not None
    assert "sk-ant-api03" not in result.error.message


def test_any_other_exception_stays_a_retryable_runtime_error() -> None:
    result = execute_tool(_call("broken", "x"), [broken])

    assert result.error is not None
    assert (result.error.type, result.error.retryable) == ("runtime_error", True)


def test_a_failure_is_metered_as_an_unbilled_failure() -> None:
    with MeterScope(RunConfig(retain_meter_events=True)) as scope:
        result = ToolGroup(find_page).execute(_call("find_page", "missing"))

    assert not result.ok
    (event,) = [e for e in scope.events() if e.kind == "tool" and e.status != "started"]
    assert (event.status, event.delivery) == ("failed", "unbilled")
    assert scope.snapshot().tool_calls == 1


def test_a_success_is_settled() -> None:
    with MeterScope(RunConfig(retain_meter_events=True)) as scope:
        assert ToolGroup(find_page).execute(_call("find_page", "Lisbon")).ok

    statuses = [e.status for e in scope.events() if e.kind == "tool"]
    assert "failed" not in statuses


# ── the result goes back marked as an error ─────────────────────────────────


def test_tool_result_marks_a_failed_call() -> None:
    assert tool_result("boom", tool_use_id="t1", is_error=True)["is_error"] is True
    assert "is_error" not in tool_result("ok", tool_use_id="t1")


async def test_run_tools_marks_failed_calls_and_only_them() -> None:
    response = Response(
        tool_calls=(
            ToolCall(id="a", name="find_page", input={"title": "missing"}),
            ToolCall(id="b", name="find_page", input={"title": "Lisbon"}),
        )
    )

    results = await run_tools(response, [find_page])
    again = run_tools_sync(response, [find_page])

    for messages in (results, again):
        assert [m.get("is_error", False) for m in messages] == [True, False]
        assert messages[0]["content"].startswith("Tool error [not_found]:")


def _turn(is_error: bool) -> list[dict[str, object]]:
    call = {"id": "t1", "name": "find_page", "input": {"title": "missing"}}
    return [
        {"role": "user", "content": "Find it."},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        tool_result(
            "Tool error [not_found]: no page",
            tool_use_id="t1",
            name="find_page",
            is_error=is_error,
        ),
    ]


@pytest.mark.parametrize("is_error", [True, False])
def test_anthropic_receives_is_error(is_error: bool) -> None:
    params = prepare(
        AnthropicProvider("claude-sonnet-4-6", "k"), _turn(is_error), max_tokens=64
    ).params

    (block,) = params["messages"][-1]["content"]
    assert block["type"] == "tool_result"
    assert block.get("is_error", False) is is_error


@pytest.mark.parametrize(("is_error", "key"), [(True, "error"), (False, "result")])
def test_gemini_receives_a_failure_under_error(is_error: bool, key: str) -> None:
    params = prepare(GeminiProvider("gemini-3-flash-preview", "k"), _turn(is_error)).params

    part = params["contents"][-1].parts[0]
    assert part.function_response.response == {key: "Tool error [not_found]: no page"}


def test_a_failure_survives_pickling_and_copying() -> None:
    failure = ToolFailure("not_found", "no page", details={"status": 404})

    for again in (pickle.loads(pickle.dumps(failure)), copy.copy(failure)):
        assert type(again) is ToolFailure
        assert again.error == failure.error and str(again) == "no page"
