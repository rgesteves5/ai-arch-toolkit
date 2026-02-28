"""Tests for LLMCompilerAgent — Plan DAG → Parallel Execute → Join."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._response import Response
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents._base import AgentConfig, AgentEvent, AgentResult
from ai_arch_toolkit.toolkit.agents._llm_compiler import (
    LLMCompilerAgent,
    LLMCompilerConfig,
    _DAGTask,
    _parse_dag,
    _ready_tasks,
    _substitute_refs,
)
from tests.agents.conftest import make_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    llm_side_effect: list[Response],
    *,
    config: AgentConfig | None = None,
    compiler: LLMCompilerConfig | None = None,
    tools: ToolGroup | None = None,
) -> LLMCompilerAgent:
    """Build an LLMCompilerAgent with mocked LLM."""

    def lookup(query: str = "") -> str:
        """Look up information."""
        return f"info:{query}"

    def calculate(expression: str = "") -> str:
        """Calculate a math expression."""
        return f"calc:{expression}"

    if tools is None:
        tools = ToolGroup(lookup, calculate)
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=llm_side_effect)
    return LLMCompilerAgent(
        llm,
        tools,
        config=config or AgentConfig(),
        compiler=compiler,
    )


# ---------------------------------------------------------------------------
# Parse helper tests
# ---------------------------------------------------------------------------


def test_parse_dag_basic():
    text = (
        "$1. Search for population of France [deps: none]\n"
        "$2. Search for population of Germany [deps: none]\n"
        "$3. Compare results [deps: $1, $2]"
    )
    dag = _parse_dag(text)
    assert len(dag) == 3
    assert dag[0].id == 1
    assert dag[0].deps == ()
    assert dag[1].id == 2
    assert dag[1].deps == ()
    assert dag[2].id == 3
    assert dag[2].deps == (1, 2)


def test_ready_tasks():
    tasks = [
        _DAGTask(id=1, description="A", deps=()),
        _DAGTask(id=2, description="B", deps=()),
        _DAGTask(id=3, description="C", deps=(1, 2)),
    ]
    ready = _ready_tasks(tasks)
    assert {t.id for t in ready} == {1, 2}

    # Mark 1 done
    tasks[0].done = True
    tasks[0].result = "result1"
    ready = _ready_tasks(tasks)
    assert {t.id for t in ready} == {2}

    # Mark 2 done
    tasks[1].done = True
    tasks[1].result = "result2"
    ready = _ready_tasks(tasks)
    assert {t.id for t in ready} == {3}


def test_substitute_refs():
    tasks = [
        _DAGTask(id=1, description="A", deps=(), result="hello", done=True),
        _DAGTask(id=2, description="B", deps=(), result="world", done=True),
    ]
    result = _substitute_refs("Combine $1 and $2", tasks)
    assert result == "Combine hello and world"


# ---------------------------------------------------------------------------
# 1. Simple DAG (2 independent tasks → join)
# ---------------------------------------------------------------------------


async def test_simple_dag_two_independent():
    plan_text = (
        "$1. Look up population of France [deps: none]\n"
        "$2. Look up population of Germany [deps: none]"
    )
    responses = [
        make_response(text=plan_text),  # Plan
        make_response(text="67 million"),  # Execute task 1 (inner ReAct)
        make_response(text="83 million"),  # Execute task 2 (inner ReAct)
        make_response(text="France: 67M, Germany: 83M"),  # Join
    ]
    agent = _make_agent(responses)
    result = await agent.run("Compare populations")

    assert result.stop_reason == "completed"
    assert result.answer == "France: 67M, Germany: 83M"


# ---------------------------------------------------------------------------
# 2. Sequential DAG (task 2 depends on task 1)
# ---------------------------------------------------------------------------


async def test_sequential_dag():
    plan_text = "$1. Look up X [deps: none]\n$2. Calculate based on $1 [deps: $1]"
    responses = [
        make_response(text=plan_text),  # Plan
        make_response(text="42"),  # Execute task 1
        make_response(text="84"),  # Execute task 2
        make_response(text="Final: 84"),  # Join
    ]
    agent = _make_agent(responses)
    result = await agent.run("Sequential task")

    assert result.stop_reason == "completed"
    assert result.answer == "Final: 84"


# ---------------------------------------------------------------------------
# 3. Parallel execution (independent tasks run via gather)
# ---------------------------------------------------------------------------


async def test_parallel_execution():
    """Independent tasks should be executed in the same batch."""
    plan_text = "$1. Task A [deps: none]\n$2. Task B [deps: none]\n$3. Task C [deps: none]"
    responses = [
        make_response(text=plan_text),  # Plan
        make_response(text="A result"),  # Execute task 1
        make_response(text="B result"),  # Execute task 2
        make_response(text="C result"),  # Execute task 3
        make_response(text="Combined"),  # Join
    ]
    agent = _make_agent(responses)
    result = await agent.run("Parallel tasks")

    assert result.stop_reason == "completed"
    assert result.answer == "Combined"


# ---------------------------------------------------------------------------
# 4. Replan on "REPLAN" response
# ---------------------------------------------------------------------------


async def test_replan_on_replan_response():
    plan_text1 = "$1. Try approach A [deps: none]"
    plan_text2 = "$1. Try approach B [deps: none]"
    responses = [
        make_response(text=plan_text1),  # Plan 1
        make_response(text="Partial result"),  # Execute task 1
        make_response(text="REPLAN\nNeed different approach"),  # Join → replan
        make_response(text=plan_text2),  # Plan 2
        make_response(text="Better result"),  # Execute task 1 v2
        make_response(text="Final answer"),  # Join → done
    ]
    agent = _make_agent(responses, compiler=LLMCompilerConfig(max_replans=2))
    result = await agent.run("Needs replan")

    assert result.stop_reason == "completed"
    assert result.answer == "Final answer"


# ---------------------------------------------------------------------------
# 5. Max replans exhausted
# ---------------------------------------------------------------------------


async def test_max_replans_exhausted():
    plan_text = "$1. Try something [deps: none]"
    responses = [
        make_response(text=plan_text),  # Plan 1
        make_response(text="Result 1"),  # Execute
        make_response(text="REPLAN\nStill wrong"),  # Join → replan
        make_response(text=plan_text),  # Plan 2
        make_response(text="Result 2"),  # Execute
        make_response(text="REPLAN\nStill wrong"),  # Join → replan
        make_response(text=plan_text),  # Plan 3
        make_response(text="Result 3"),  # Execute
        make_response(text="REPLAN\nGave up"),  # Join → replan (exhausted)
    ]
    agent = _make_agent(responses, compiler=LLMCompilerConfig(max_replans=2))
    result = await agent.run("Exhaust replans")

    assert result.stop_reason == "max_iterations"


# ---------------------------------------------------------------------------
# 6. Empty DAG (no parseable tasks → straight to join)
# ---------------------------------------------------------------------------


async def test_empty_dag():
    responses = [
        make_response(text="I'll just answer directly."),  # Plan (no DAG tasks)
        make_response(text="Direct answer"),  # Join
    ]
    agent = _make_agent(responses)
    result = await agent.run("Simple question")

    assert result.stop_reason == "completed"
    assert result.answer == "Direct answer"


# ---------------------------------------------------------------------------
# 7. Event sequence continuity
# ---------------------------------------------------------------------------


async def test_event_sequence_continuity():
    plan_text = "$1. Task A [deps: none]\n$2. Task B [deps: $1]"
    responses = [
        make_response(text=plan_text),
        make_response(text="A done"),
        make_response(text="B done"),
        make_response(text="Done"),
    ]
    events: list[AgentEvent] = []
    config = AgentConfig(on_event=events.append)
    agent = _make_agent(responses, config=config)
    await agent.run("Test events")

    steps = [e.step for e in events]
    assert steps == sorted(steps)
    # At minimum: plan, exec1, exec2, join = 4 distinct steps
    assert len(set(steps)) >= 4


# ---------------------------------------------------------------------------
# 8. run_sync works
# ---------------------------------------------------------------------------


def test_run_sync():
    plan_text = "$1. Do something [deps: none]"
    responses = [
        make_response(text=plan_text),
        make_response(text="Result"),
        make_response(text="Sync answer"),
    ]
    agent = _make_agent(responses)
    result = agent.run_sync("Sync test")

    assert isinstance(result, AgentResult)
    assert result.answer == "Sync answer"
    assert result.stop_reason == "completed"


# ---------------------------------------------------------------------------
# 9. Invalid/unresolvable dependencies (deadlock guard)
# ---------------------------------------------------------------------------


async def test_invalid_deps_skipped():
    """Tasks referencing nonexistent IDs are marked failed and skipped."""
    plan_text = "$1. Do something [deps: none]\n$2. Depends on nonexistent [deps: $99]"
    responses = [
        make_response(text=plan_text),  # Plan
        make_response(text="Task 1 done"),  # Execute task 1
        make_response(text="Final answer"),  # Join
    ]
    agent = _make_agent(responses)
    result = await agent.run("Test invalid deps")

    assert result.stop_reason == "completed"
    assert result.answer == "Final answer"


# ---------------------------------------------------------------------------
# 10. Failed task propagates to dependents
# ---------------------------------------------------------------------------


async def test_failed_task_propagates_to_dependents():
    """When a task fails, its dependents are auto-skipped."""
    plan_text = "$1. Do something risky [deps: none]\n$2. Depends on risky task [deps: $1]"
    call_count = {"n": 0}
    plan_response = make_response(text=plan_text)

    async def _side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        if idx == 0:
            return plan_response  # Plan
        if idx == 1:
            raise RuntimeError("Inner ReAct LLM failed")  # Task 1 fails
        # Join — task 2 should be auto-skipped
        return make_response(text="Handled failure gracefully")

    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=_side_effect)

    def lookup(query: str = "") -> str:
        """Look up information."""
        return f"info:{query}"

    tools = ToolGroup(lookup)
    agent = LLMCompilerAgent(llm, tools, config=AgentConfig())
    result = await agent.run("Test failure propagation")

    assert result.stop_reason == "completed"
    assert result.answer == "Handled failure gracefully"
    # Should be 3 calls: plan, failed task 1, join (task 2 auto-skipped)
    assert call_count["n"] == 3


# ---------------------------------------------------------------------------
# 11. Tool use within DAG tasks
# ---------------------------------------------------------------------------


async def test_chained_failure_propagation():
    """Failure cascades through multi-level dependency chains: A→B→C."""
    plan_text = (
        "$1. Do something risky [deps: none]\n"
        "$2. Depends on task 1 [deps: $1]\n"
        "$3. Depends on task 2 [deps: $2]"
    )
    call_count = {"n": 0}
    plan_response = make_response(text=plan_text)

    async def _side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        if idx == 0:
            return plan_response  # Plan
        if idx == 1:
            raise RuntimeError("Task 1 failed")  # Task 1 fails
        # Join — tasks 2 and 3 should both be auto-skipped
        return make_response(text="Handled chain failure")

    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=_side_effect)

    def lookup(query: str = "") -> str:
        """Look up information."""
        return f"info:{query}"

    tools = ToolGroup(lookup)
    agent = LLMCompilerAgent(llm, tools, config=AgentConfig())
    result = await agent.run("Test chain failure")

    assert result.stop_reason == "completed"
    assert result.answer == "Handled chain failure"
    # Should be 3 calls: plan, failed task 1, join (tasks 2+3 auto-skipped)
    assert call_count["n"] == 3


# ---------------------------------------------------------------------------
# 12. Tool use within DAG tasks
# ---------------------------------------------------------------------------


async def test_tool_use_in_dag_task():
    """DAG tasks can execute tools via inner ReAct."""
    from tests.agents.conftest import make_tool_call

    plan_text = "$1. Look up the answer [deps: none]"
    tc = make_tool_call(name="lookup", input={"query": "test"}, id="tc_1")
    responses = [
        make_response(text=plan_text),  # Plan
        make_response(text="", tool_calls=(tc,)),  # Execute: tool call
        make_response(text="Found the answer"),  # Execute: after tool
        make_response(text="Final answer from tool"),  # Join
    ]
    agent = _make_agent(responses)
    result = await agent.run("Use tools")

    assert result.stop_reason == "completed"
    assert result.answer == "Final answer from tool"
