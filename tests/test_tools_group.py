"""Tests for _tools/_group.py — ToolGroup with governance."""

from __future__ import annotations

import asyncio
import copy
import functools
import threading

import pytest

from ai_arch_toolkit.core._response import Response, ToolCall
from ai_arch_toolkit.core._server_tools import code_execution, web_search
from ai_arch_toolkit.core._tools import prepare_tools
from ai_arch_toolkit.core._tools._approval import ApprovalDecision
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._executor import async_execute_tool, execute_tool
from ai_arch_toolkit.core._tools._governance import DangerousToolGate, DryRunGate
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit import run_tools, run_tools_sync


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"


@tool
def search(query: str) -> dict:
    """Search the web."""
    return {"results": [query]}


@tool
async def async_fetch(url: str) -> str:
    """Fetch a URL."""
    return f"content_of_{url}"


@tool
def explode() -> str:
    """Raise a runtime error."""
    raise RuntimeError("boom")


@tool(capability="shell", risk_level="critical", requires_approval=True)
def dangerous_echo(command: str) -> str:
    """Echo a dangerous command."""
    return command


def scale(factor: float, value: int) -> float:
    """Scale a value."""
    return factor * value


@tool(requires_approval=True)
def guarded(command: str = "noop") -> str:
    """Run a command that needs approval."""
    return command


def plain_function(x: int) -> int:
    """Double a number."""
    return x * 2


class TestToolGroupBasics:
    def test_from_decorated(self):
        group = ToolGroup(get_weather, search)
        assert len(group) == 2
        assert "get_weather" in group
        assert "search" in group

    def test_definitions_are_provider_safe(self):
        group = ToolGroup(dangerous_echo)
        defs = group.definitions
        assert len(defs) == 1
        assert defs[0]["name"] == "dangerous_echo"
        assert "input_schema" in defs[0]
        # No runtime governance metadata leaks into provider schemas.
        assert "requires_approval" not in defs[0]
        assert "risk_level" not in defs[0]
        assert "capability" not in defs[0]

    def test_runtime_definitions_carry_policy(self):
        group = ToolGroup(dangerous_echo)
        rt = group.runtime_definitions
        assert rt[0].policy.requires_approval is True
        assert rt[0].policy.risk_level == "critical"

    def test_add(self):
        group = ToolGroup()
        assert len(group) == 0
        group.add(get_weather)
        assert len(group) == 1

    def test_contains(self):
        group = ToolGroup(get_weather)
        assert "get_weather" in group
        assert "missing" not in group

    def test_repr(self):
        group = ToolGroup(get_weather, search)
        r = repr(group)
        assert "ToolGroup" in r
        assert "get_weather" in r
        assert "search" in r

    def test_plain_function_auto_inferred(self):
        group = ToolGroup(plain_function)
        assert "plain_function" in group
        defs = group.definitions
        assert defs[0]["name"] == "plain_function"
        assert defs[0]["description"] == "Double a number."


class TestOneToolPerName:
    """Two tools under one name leave the model unable to choose: the group refuses the second."""

    def test_another_tool_under_a_name_the_group_has_is_an_error(self):
        @tool(name="get_weather")
        def other_weather(city: str) -> str:
            """Another weather tool."""
            return f"Rain in {city}"

        group = ToolGroup(get_weather)

        with pytest.raises(ValueError, match="'get_weather'"):
            group.add(other_weather)
        kept = group.execute(ToolCall(id="tc_1", name="get_weather", input={"city": "Porto"}))
        assert kept.value == "Sunny in Porto"

    def test_the_constructor_refuses_two_callables_with_one_name(self):
        with pytest.raises(ValueError, match="'scale'"):
            ToolGroup(functools.partial(scale, 2.0), functools.partial(scale, 3.0))

    def test_the_same_tool_twice_is_one_tool(self):
        group = ToolGroup(get_weather, get_weather)

        assert len(group) == 1

    def test_the_same_method_read_twice_is_one_tool(self):
        class Weather:
            def forecast(self, city: str) -> str:
                """The forecast for a city."""
                return f"Sun in {city}"

        weather = Weather()
        group = ToolGroup(weather.forecast, weather.forecast)

        assert len(group) == 1
        result = group.execute(ToolCall(id="tc_1", name="forecast", input={"city": "Porto"}))
        assert result.value == "Sun in Porto"


class TestChangingTheGroup:
    """``add(replace=)`` and ``remove()`` write a new table: what is running keeps the old one."""

    def test_replace_puts_the_new_tool_under_the_name(self):
        @tool(name="get_weather")
        def rainy_weather(city: str) -> str:
            """Another weather tool."""
            return f"Rain in {city}"

        group = ToolGroup(get_weather, search)
        group.add(rainy_weather, replace=True)

        result = group.execute(ToolCall(id="tc_1", name="get_weather", input={"city": "Porto"}))
        assert result.value == "Rain in Porto"
        assert group.tools == [rainy_weather, search]

    def test_replace_adds_a_tool_the_group_does_not_hold(self):
        group = ToolGroup(get_weather)
        group.add(search, replace=True)

        assert group.tools == [get_weather, search]

    def test_remove_returns_the_definition_and_the_name_becomes_unknown(self):
        group = ToolGroup(get_weather, search)

        removed = group.remove("get_weather")

        assert removed is get_weather.__tool_definition__
        assert "get_weather" not in group and len(group) == 1
        result = group.execute(ToolCall(id="tc_1", name="get_weather", input={"city": "Porto"}))
        assert result.error is not None and result.error.type == "unknown_tool"

    def test_removing_a_name_the_group_does_not_hold_is_a_key_error(self):
        group = ToolGroup(get_weather)

        with pytest.raises(KeyError, match="'search'"):
            group.remove("search")
        assert group.tools == [get_weather]

    async def test_a_running_call_finishes_with_the_tool_it_started_with(self):
        started, release = asyncio.Event(), asyncio.Event()

        @tool
        async def wait_for_release() -> str:
            """Wait until the test releases the call."""
            started.set()
            await release.wait()
            return "finished"

        group = ToolGroup(wait_for_release)
        running = asyncio.create_task(
            group.async_execute(ToolCall(id="tc_1", name="wait_for_release", input={}))
        )
        await started.wait()

        group.remove("wait_for_release")
        release.set()
        result = await running

        assert result.ok and result.value == "finished"
        assert "wait_for_release" not in group

    def test_the_call_budget_does_not_restart_when_the_group_changes(self):
        group = ToolGroup(get_weather, max_calls=1)
        assert group.execute(ToolCall(id="t1", name="get_weather", input={"city": "a"})).ok

        group.add(search)
        group.remove("get_weather")
        blocked = group.execute(ToolCall(id="t2", name="search", input={"query": "q"}))

        assert blocked.error is not None and blocked.error.type == "max_calls_exceeded"

    def test_definitions_read_before_a_change_stay_as_they_were(self):
        group = ToolGroup(get_weather)
        before = group.runtime_definitions

        group.add(search)
        group.remove("get_weather")

        assert [d.schema.name for d in before] == ["get_weather"]
        assert [d["name"] for d in group.definitions] == ["search"]

    def test_a_copied_group_changes_on_its_own(self):
        group = ToolGroup(get_weather)

        copied = copy.deepcopy(group)
        copied.add(search)

        assert group.tools == [get_weather]
        assert [d["name"] for d in copied.definitions] == ["get_weather", "search"]

    def test_threads_adding_at_once_lose_no_tool(self):
        def make(index: int):
            @tool(name=f"tool_{index}")
            def numbered() -> str:
                """A numbered tool."""
                return str(index)

            return numbered

        made = [make(i) for i in range(64)]
        group = ToolGroup()
        barrier = threading.Barrier(len(made))

        def add(fn) -> None:
            barrier.wait()
            group.add(fn)

        threads = [threading.Thread(target=add, args=(fn,)) for fn in made]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(group) == len(made)


class TestOneToolPerNameAcrossListsAndGroups:
    """D62 (C02.8): a name the model could not choose by is refused before sending or running."""

    def test_prepare_tools_refuses_two_tools_with_one_name_in_a_list(self):
        @tool(name="get_weather")
        def other_weather(city: str) -> str:
            """Another weather tool."""
            return city

        with pytest.raises(ValueError, match="'get_weather'"):
            prepare_tools([get_weather, other_weather])

    def test_prepare_tools_refuses_one_name_in_two_groups(self):
        @tool(name="search")
        def other_search(query: str) -> str:
            """Another search."""
            return query

        with pytest.raises(ValueError, match="'search'"):
            prepare_tools([ToolGroup(search), ToolGroup(other_search)])

    def test_prepare_tools_refuses_a_dict_and_a_tool_with_one_name(self):
        wire = {"name": "search", "description": "", "input_schema": {"type": "object"}}

        with pytest.raises(ValueError, match="'search'"):
            prepare_tools([wire, search])

    def test_the_same_tool_met_again_is_sent_once(self):
        wire = {"name": "raw", "description": "", "input_schema": {"type": "object"}}

        sent = prepare_tools(
            [get_weather, ToolGroup(get_weather, search), get_weather, wire, dict(wire)]
        )

        assert sent is not None
        assert [d["name"] for d in sent] == ["get_weather", "search", "raw"]

    def test_execute_tool_refuses_two_tools_with_one_name_before_running(self):
        ran: list[str] = []

        @tool(name="get_weather")
        def recording_weather(city: str) -> str:
            """Another weather tool."""
            ran.append(city)
            return city

        with pytest.raises(ValueError, match="'get_weather'"):
            execute_tool(
                ToolCall(id="tc_1", name="get_weather", input={"city": "Porto"}),
                [recording_weather, get_weather],
            )
        assert ran == []

    async def test_run_tools_refuses_a_plain_function_named_like_a_tool_before_running(self):
        ran: list[str] = []

        def plain(query: str) -> str:
            """A plain function that will carry the decorated search tool's name."""
            ran.append(query)
            return query

        plain.__name__ = "search"
        response = Response(tool_calls=(ToolCall(id="tc_1", name="search", input={"query": "q"}),))

        with pytest.raises(ValueError, match="'search'"):
            await run_tools(response, [plain, search])
        assert ran == []

    def test_the_same_tool_twice_in_a_list_runs_once(self):
        result = execute_tool(
            ToolCall(id="tc_1", name="get_weather", input={"city": "Porto"}),
            [get_weather, get_weather],
        )

        assert result.value == "Sunny in Porto"

    async def test_a_list_llm_complete_takes_runs_without_a_false_clash(self):
        # Server tools, wire-form dicts and groups have no name of the app's: two of a kind never
        # clash by their type's name.
        tools = [
            get_weather,
            web_search(),
            code_execution(),
            {"_server_tool": True, "type": "web_search"},
            {"_server_tool": True, "type": "code_execution"},
            ToolGroup(search),
            ToolGroup(async_fetch),
        ]
        call = ToolCall(id="tc_1", name="get_weather", input={"city": "Porto"})

        assert prepare_tools(tools) is not None
        assert execute_tool(call, tools).value == "Sunny in Porto"
        assert (await async_execute_tool(call, tools)).value == "Sunny in Porto"
        response = Response(tool_calls=(call,))
        assert [r["content"] for r in run_tools_sync(response, tools)] == ["Sunny in Porto"]
        assert [r["content"] for r in await run_tools(response, tools)] == ["Sunny in Porto"]

    def test_the_executor_refuses_the_dict_and_tool_with_one_name_that_prepare_tools_refuses(
        self,
    ):
        wire = {"name": "search", "description": "", "input_schema": {"type": "object"}}
        call = ToolCall(id="tc_1", name="search", input={"query": "q"})

        with pytest.raises(ValueError, match="'search'"):
            prepare_tools([wire, search])
        with pytest.raises(ValueError, match="'search'"):
            execute_tool(call, [wire, search])

    def test_a_call_to_a_tool_dict_is_an_unknown_tool(self):
        wire = {"name": "raw", "description": "", "input_schema": {"type": "object"}}

        result = execute_tool(ToolCall(id="tc_1", name="raw", input={}), [wire, get_weather])

        assert result.error is not None and result.error.type == "unknown_tool"


class TestWrappedTools:
    """``functools.wraps`` copies a tool's definition onto its wrapper, and the wrapper runs."""

    def test_a_group_runs_the_wrapper(self):
        seen: list[str] = []

        @functools.wraps(get_weather)
        def logged(*args, **kwargs):
            seen.append("wrapper")
            return get_weather(*args, **kwargs)

        group = ToolGroup(logged)
        result = group.execute(ToolCall(id="tc_1", name="get_weather", input={"city": "Porto"}))

        assert result.ok is True, result.to_model_text()
        assert result.value == "Sunny in Porto"
        assert seen == ["wrapper"]
        assert group.tools == [logged]

    def test_execute_tool_runs_the_wrapper(self):
        @functools.wraps(get_weather)
        def shouting(*args, **kwargs):
            return get_weather(*args, **kwargs).upper()

        result = execute_tool(
            ToolCall(id="tc_1", name="get_weather", input={"city": "Porto"}), [shouting]
        )

        assert result.value == "SUNNY IN PORTO"

    def test_the_wrapper_and_the_tool_it_wraps_are_two_tools_with_one_name(self):
        @functools.wraps(get_weather)
        def logged(*args, **kwargs):
            return get_weather(*args, **kwargs)

        with pytest.raises(ValueError, match="'get_weather'"):
            ToolGroup(get_weather, logged)


class TestRejectsNonLocalTools:
    """A group only holds locally executable callables; anything else fails loudly."""

    def test_server_tool_in_constructor_raises_type_error(self):
        with pytest.raises(TypeError, match=r"tools=\[") as excinfo:
            ToolGroup(get_weather, web_search())  # type: ignore[arg-type]
        message = str(excinfo.value)
        assert "'web_search'" in message
        assert "provider" in message

    def test_server_tool_via_add_raises_and_leaves_group_unchanged(self):
        group = ToolGroup(get_weather)
        with pytest.raises(TypeError, match=r"tools=\["):
            group.add(code_execution())  # type: ignore[arg-type]
        assert len(group) == 1
        assert group.definitions[0]["name"] == "get_weather"

    def test_non_callable_raises_type_error(self):
        with pytest.raises(TypeError, match="callable"):
            ToolGroup(42)  # type: ignore[arg-type]


class TestExecute:
    def test_success(self):
        group = ToolGroup(get_weather, search)
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        result = group.execute(tc)
        assert result.ok is True
        assert result.value == "Sunny in NYC"

    def test_json_value(self):
        group = ToolGroup(search)
        tc = ToolCall(id="tc_1", name="search", input={"query": "test"})
        result = group.execute(tc)
        assert result.to_model_text() == '{"results": ["test"]}'

    def test_unknown(self):
        group = ToolGroup(get_weather)
        tc = ToolCall(id="tc_1", name="missing", input={})
        result = group.execute(tc)
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "unknown_tool"

    def test_validation_error(self):
        group = ToolGroup(get_weather)
        tc = ToolCall(id="tc_1", name="get_weather", input={})
        result = group.execute(tc)
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "validation_error"

    def test_runtime_error(self):
        group = ToolGroup(explode)
        tc = ToolCall(id="tc_1", name="explode", input={})
        result = group.execute(tc)
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "runtime_error"

    def test_plain_function(self):
        group = ToolGroup(plain_function)
        tc = ToolCall(id="tc_1", name="plain_function", input={"x": 5})
        result = group.execute(tc)
        assert result.value == 10

    async def test_async_sync_fn(self):
        group = ToolGroup(get_weather)
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "LA"})
        result = await group.async_execute(tc)
        assert result.value == "Sunny in LA"

    async def test_async_async_fn(self):
        group = ToolGroup(async_fetch)
        tc = ToolCall(id="tc_1", name="async_fetch", input={"url": "http://example.com"})
        result = await group.async_execute(tc)
        assert result.value == "content_of_http://example.com"


class TestApproval:
    def test_missing_handler_denies(self):
        group = ToolGroup(dangerous_echo)
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": "rm -rf /"})
        result = group.execute(tc)
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "approval_denied"

    def test_approved_with_audit(self):
        group = ToolGroup(
            dangerous_echo,
            approval_handler=lambda _: ApprovalDecision.approve(reviewer="human"),
        )
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": "echo ok"})
        result = group.execute(tc)
        assert result.ok is True
        assert result.value == "echo ok"
        assert result.metadata["audit"]["approval"]["decision"]["reviewer"] == "human"

    def test_approval_with_empty_modified_args_calls_without_arguments(self):
        group = ToolGroup(
            guarded,
            approval_handler=lambda _: ApprovalDecision.approve(modified_args={}),
        )
        tc = ToolCall(id="tc_1", name="guarded", input={"command": "rm -rf /"})

        result = group.execute(tc)

        assert result.ok is True
        assert result.value == "noop"

    async def test_async_approved_modified_args(self):
        async def approve(_request):
            return ApprovalDecision.approve(modified_args={"command": "echo safe"})

        group = ToolGroup(dangerous_echo, approval_handler=approve)
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": "rm -rf /"})
        result = await group.async_execute(tc)
        assert result.ok is True
        assert result.value == "echo safe"

    async def test_handler_preserved_when_composed_with_gates(self):
        """Regression: composing governance gates must not drop the handler."""

        async def approve(_request):
            return ApprovalDecision.approve(reviewer="human")

        group = ToolGroup(
            dangerous_echo,
            approval_handler=approve,
            gates=(DangerousToolGate(blocked={"other"}, allow=False),),
        )
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": "echo ok"})
        result = await group.async_execute(tc)
        assert result.ok is True
        assert result.value == "echo ok"


class TestGovernanceGates:
    def test_dangerous_blocked(self):
        group = ToolGroup(
            get_weather,
            gates=(DangerousToolGate(blocked={"get_weather"}, allow=False),),
        )
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        result = group.execute(tc)
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "dangerous_tool_blocked"

    def test_the_block_reads_as_a_sentence_for_a_person(self):
        """The model repeats the block to the person: it names no command-line flag."""
        group = ToolGroup(get_weather, gates=(DangerousToolGate(blocked={"get_weather"}),))

        result = group.execute(ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"}))

        assert result.error is not None
        message = result.error.message
        assert "--" not in message
        assert message.startswith("The tool 'get_weather' did not run")
        assert "dangerous" in message

    def test_dangerous_allowed(self):
        group = ToolGroup(
            get_weather,
            gates=(DangerousToolGate(blocked={"get_weather"}, allow=True),),
        )
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        result = group.execute(tc)
        assert result.ok is True

    def test_dry_run_does_not_execute(self):
        calls: list[str] = []

        @tool
        def record(text: str) -> str:
            """Record a call."""
            calls.append(text)
            return text

        group = ToolGroup(record, gates=(DryRunGate(dry_run=True),))
        tc = ToolCall(id="tc_1", name="record", input={"text": "hi"})
        result = group.execute(tc)
        assert result.ok is True
        assert result.metadata["governance"]["outcome"] == "dry_run"
        assert result.metadata["governance"]["executed"] is False
        assert calls == []
        # The model-facing text never includes raw arguments.
        assert "hi" not in result.to_model_text()


class TestCallBudget:
    def test_max_calls_blocks_after_limit(self):
        group = ToolGroup(get_weather, max_calls=1)
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        first = group.execute(tc)
        second = group.execute(tc)
        assert first.ok is True
        assert second.ok is False
        assert second.error is not None
        assert second.error.type == "max_calls_exceeded"

    def test_blocked_call_does_not_consume_budget(self):
        # dry-run short-circuits before the budget commit, so it never counts.
        group = ToolGroup(get_weather, max_calls=1, gates=(DryRunGate(dry_run=True),))
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        first = group.execute(tc)
        second = group.execute(tc)
        assert first.metadata["governance"]["outcome"] == "dry_run"
        assert second.metadata["governance"]["outcome"] == "dry_run"

    def test_reset_restores_budget(self):
        group = ToolGroup(get_weather, max_calls=1)
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        group.execute(tc)
        assert group.execute(tc).ok is False
        group.reset()
        assert group.execute(tc).ok is True

    def test_approval_denied_does_not_consume_budget(self):
        # No handler → dangerous_echo is denied by approval before the budget commit.
        group = ToolGroup(get_weather, dangerous_echo, max_calls=1)
        denied = group.execute(
            ToolCall(id="t1", name="dangerous_echo", input={"command": "echo x"})
        )
        assert denied.error is not None
        assert denied.error.type == "approval_denied"
        # The denied call must not have consumed the single budget slot.
        allowed = group.execute(ToolCall(id="t2", name="get_weather", input={"city": "NYC"}))
        assert allowed.ok is True

    async def test_max_calls_atomic_under_gather(self):
        """Concurrent calls must not exceed the budget (atomic reserve)."""
        limit = 3
        group = ToolGroup(async_fetch, max_calls=limit)
        tc = ToolCall(id="tc", name="async_fetch", input={"url": "x"})
        results = await asyncio.gather(*[group.async_execute(tc) for _ in range(limit + 5)])
        executed = [r for r in results if r.ok]
        blocked = [r for r in results if not r.ok]
        assert len(executed) == limit
        assert all(r.error.type == "max_calls_exceeded" for r in blocked if r.error)


class TestPartialTools:
    def test_a_partial_runs_under_the_wrapped_function_name(self):
        group = ToolGroup(functools.partial(scale, 2.0))

        result = group.execute(ToolCall(id="tc_1", name="scale", input={"value": 3}))

        assert result.ok is True, result.to_model_text()
        assert result.value == 6.0
