"""Output and time bounds the governed executor imposes on every tool (D17).

``ToolRuntimePolicy.max_output_chars`` (200 000 by default) cuts what the model receives and marks
the cut; ``timeout_s`` (120 by default) stops the wait. ``@tool(...)`` sets them per tool and a
``ToolGroup`` can only tighten them.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time

import pytest

from ai_arch_toolkit.core._metering._scope import MeterScope
from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._definition import ToolRuntimePolicy
from ai_arch_toolkit.core._tools._executor import async_execute_tool, execute_tool
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.core._tools._result import ToolResult

FIVE_MB = 5_000_000


def call(name: str, **args: object) -> ToolCall:
    return ToolCall(id="call_1", name=name, input=args)


@tool
def huge() -> str:
    """Return five megabytes of text."""
    return "x" * FIVE_MB


@tool
def huge_rows() -> list[dict[str, str]]:
    """Return a structured value whose JSON is several megabytes."""
    return [{"row": "y" * 100}] * 50_000


@tool
def huge_failure() -> ToolResult:
    """Fail with a five-megabyte message."""
    return ToolResult.failure("upstream", "z" * FIVE_MB)


@tool(max_output_chars=50)
def terse() -> str:
    """Declare a small output bound."""
    return "t" * 1000


@tool(max_output_chars=None)
def unbounded() -> str:
    """Declare no output bound."""
    return "u" * 300_000


class TestOutputBound:
    def test_a_five_megabyte_result_is_cut_at_the_default_limit_and_marked(self) -> None:
        result = execute_tool(call("huge"), [huge])

        text = result.to_model_text()
        assert result.ok
        assert text.startswith("x" * 200_000)
        assert text[200_000:] == "\n\n[Output truncated: kept 200000 of 5000000 characters.]"
        assert result.metadata["truncated"] == {"chars": FIVE_MB, "kept": 200_000}

    async def test_the_async_path_cuts_too(self) -> None:
        result = await async_execute_tool(call("huge"), [huge])

        assert len(result.to_model_text()) < 200_100
        assert result.metadata["truncated"]["kept"] == 200_000

    def test_a_structured_value_is_cut_as_the_json_the_model_would_read(self) -> None:
        result = execute_tool(call("huge_rows"), [huge_rows])

        assert isinstance(result.value, str)
        assert result.value.startswith('[{"row": "yyy')
        assert result.metadata["truncated"]["kept"] == 200_000

    def test_a_long_error_message_is_cut(self) -> None:
        result = execute_tool(call("huge_failure"), [huge_failure])

        assert not result.ok
        assert result.error is not None
        assert result.error.type == "upstream"
        assert len(result.to_model_text()) < 200_100
        assert result.metadata["truncated"]["chars"] > FIVE_MB

    def test_a_tool_declares_its_own_bound_or_none(self) -> None:
        assert execute_tool(call("terse"), [terse]).value.startswith("t" * 50 + "\n\n[Output")
        assert execute_tool(call("unbounded"), [unbounded]).value == "u" * 300_000

    def test_a_group_tightens_and_never_widens(self) -> None:
        group = ToolGroup(huge, terse, max_output_chars=100)

        assert group.execute(call("huge")).metadata["truncated"]["kept"] == 100
        assert group.execute(call("terse")).metadata["truncated"]["kept"] == 50

    def test_a_short_result_is_untouched(self) -> None:
        @tool
        def short() -> str:
            """Return a little text."""
            return "ok"

        result = execute_tool(call("short"), [short])

        assert result.value == "ok"
        assert "truncated" not in result.metadata


cancelled: list[str] = []


@tool(timeout_s=0.05)
async def slow_async() -> str:
    """Wait far longer than the deadline."""
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        cancelled.append("slow_async")
        raise
    return "late"


@tool(timeout_s=0.05)
def slow_sync() -> str:
    """Block far longer than the deadline."""
    time.sleep(2)
    return "late"


@tool
def slow_default() -> str:
    """Take a moment under the default deadline."""
    time.sleep(0.3)
    return "done"


@tool(timeout_s=None)
def no_deadline() -> str:
    """Take a moment with no deadline."""
    time.sleep(0.2)
    return "done"


@tool
def own_timeout() -> str:
    """Fail with the tool's own TimeoutError."""
    raise TimeoutError("upstream read timed out")


def _assert_timed_out(result: ToolResult, started: float, deadline: float = 0.05) -> None:
    assert not result.ok
    assert result.error is not None
    assert result.error.type == "timeout"
    assert result.error.retryable
    assert f"within {deadline:g}s" in result.error.message
    assert time.monotonic() - started < 1.0


class TestTimeout:
    async def test_an_async_tool_is_cancelled_at_the_deadline(self) -> None:
        cancelled.clear()
        started = time.monotonic()

        result = await async_execute_tool(call("slow_async"), [slow_async])

        _assert_timed_out(result, started)
        assert cancelled == ["slow_async"]

    async def test_the_async_path_stops_waiting_for_a_blocked_sync_tool(self) -> None:
        started = time.monotonic()

        _assert_timed_out(await async_execute_tool(call("slow_sync"), [slow_sync]), started)

    def test_the_sync_path_stops_waiting_for_a_blocked_sync_tool(self) -> None:
        started = time.monotonic()

        _assert_timed_out(execute_tool(call("slow_sync"), [slow_sync]), started)

    def test_the_sync_path_bounds_an_async_tool(self) -> None:
        started = time.monotonic()

        _assert_timed_out(execute_tool(call("slow_async"), [slow_async]), started)

    def test_a_group_deadline_tightens_the_default(self) -> None:
        started = time.monotonic()

        result = ToolGroup(slow_default, timeout_s=0.05).execute(call("slow_default"))

        _assert_timed_out(result, started)

    def test_no_deadline_waits_and_a_group_can_still_set_one(self) -> None:
        assert execute_tool(call("no_deadline"), [no_deadline]).value == "done"
        started = time.monotonic()
        result = ToolGroup(no_deadline, timeout_s=0.05).execute(call("no_deadline"))
        _assert_timed_out(result, started)

    def test_the_tools_own_timeout_error_is_a_runtime_error(self) -> None:
        result = execute_tool(call("own_timeout"), [own_timeout])

        assert result.error is not None
        assert result.error.type == "runtime_error"
        assert "upstream read timed out" in result.error.message

    async def test_a_timed_out_tool_releases_its_meter_operation(self) -> None:
        with MeterScope() as scope:
            await async_execute_tool(call("slow_sync"), [slow_sync])

        snap = scope.snapshot()
        assert snap.tool_calls == 1
        assert snap.out_tool_calls == 0


marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker", default="unset")


@tool
def where_am_i() -> str:
    """Report the thread the tool runs in and the caller's context it sees."""
    thread = threading.current_thread()
    return f"{thread is threading.main_thread()}|{thread.daemon}|{marker.get()}"


class TestThread:
    def test_a_sync_tool_runs_in_a_daemon_thread_with_the_callers_context(self) -> None:
        token = marker.set("from the caller")
        try:
            result = execute_tool(call("where_am_i"), [where_am_i])
        finally:
            marker.reset(token)

        assert result.value == "False|True|from the caller"


class TestDeclaration:
    @pytest.mark.parametrize(
        "bounds", [{"max_output_chars": 0}, {"max_output_chars": -5}, {"timeout_s": 0}]
    )
    def test_a_bound_must_be_positive(self, bounds: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="positive"):
            ToolRuntimePolicy(**bounds)
        with pytest.raises(ValueError, match="positive"):
            ToolGroup(**bounds)
