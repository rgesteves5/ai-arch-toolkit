"""Tests for LLMCompilerAgent."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from unittest.mock import AsyncMock, MagicMock

from ai_arch_toolkit.agents._base import AgentConfig, AgentEvent
from ai_arch_toolkit.agents._compiler import LLMCompilerAgent
from ai_arch_toolkit.llm._types import Response, Tool, Usage
from ai_arch_toolkit.tools._registry import ToolRegistry


def _make_client(responses: list[Response]) -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock(side_effect=responses)
    return client


def _make_async_client(responses: list[Response]) -> MagicMock:
    client = MagicMock()
    client.chat = AsyncMock(side_effect=responses)
    return client


def _make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        "search",
        lambda input: f"found '{input}'",
        Tool(name="search", description="", parameters={}),
    )
    reg.register(
        "summarize",
        lambda input: f"summary of '{input}'",
        Tool(name="summarize", description="", parameters={}),
    )
    return reg


def test_compiler_sequential_dag():
    """Sequential DAG executes tasks in dependency order."""
    dag_json = json.dumps(
        [
            {"id": "1", "tool": "search", "args": {"input": "query"}, "deps": []},
            {"id": "2", "tool": "summarize", "args": {"input": "$1"}, "deps": ["1"]},
        ]
    )
    responses = [
        Response(text=dag_json, usage=Usage()),
        Response(text="Final answer.", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = _make_registry()
    agent = LLMCompilerAgent(client, reg)
    result = agent.run("Search and summarize")

    assert result.answer == "Final answer."
    assert client.chat.call_count == 2  # plan + join


def test_compiler_parallel_dag():
    """Independent tasks execute (potentially in parallel)."""
    dag_json = json.dumps(
        [
            {"id": "1", "tool": "search", "args": {"input": "A"}, "deps": []},
            {"id": "2", "tool": "search", "args": {"input": "B"}, "deps": []},
        ]
    )
    responses = [
        Response(text=dag_json, usage=Usage()),
        Response(text="Combined answer.", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = _make_registry()
    agent = LLMCompilerAgent(client, reg)
    result = agent.run("Search A and B")

    assert result.answer == "Combined answer."


def test_compiler_dependency_substitution():
    """$N references are replaced with actual results."""
    dag_json = json.dumps(
        [
            {"id": "1", "tool": "search", "args": {"input": "topic"}, "deps": []},
            {"id": "2", "tool": "summarize", "args": {"input": "$1"}, "deps": ["1"]},
        ]
    )
    responses = [
        Response(text=dag_json, usage=Usage()),
        Response(text="done", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = _make_registry()
    agent = LLMCompilerAgent(client, reg)
    result = agent.run("test")

    # The last step should have tool results
    tool_results = result.steps[-1].tool_results
    assert len(tool_results) == 2
    # Task 2 should have received task 1's output via $1
    assert "found 'topic'" in tool_results[1].content


def test_compiler_joiner_synthesis():
    """Joiner LLM call receives all task results."""
    dag_json = json.dumps(
        [
            {"id": "1", "tool": "search", "args": {"input": "x"}, "deps": []},
        ]
    )
    responses = [
        Response(text=dag_json, usage=Usage()),
        Response(text="Synthesized.", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = _make_registry()
    agent = LLMCompilerAgent(client, reg)
    agent.run("test")

    # Verify joiner call includes task results
    join_call = client.chat.call_args_list[-1]
    join_msg = join_call[0][0][0].content
    assert "found 'x'" in join_msg


def test_compiler_events():
    """Events include plan_created, tool_call, tool_result."""
    events: list[AgentEvent] = []
    dag_json = json.dumps(
        [
            {"id": "1", "tool": "search", "args": {"input": "q"}, "deps": []},
        ]
    )
    responses = [
        Response(text=dag_json, usage=Usage()),
        Response(text="done", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = _make_registry()
    config = AgentConfig(on_event=events.append)
    agent = LLMCompilerAgent(client, reg, config=config)
    agent.run("test")

    types = [e.type for e in events]
    assert "plan_created" in types
    assert "tool_call" in types
    assert "tool_result" in types


def test_compiler_async():
    """Async compiler works correctly."""
    dag_json = json.dumps(
        [
            {"id": "1", "tool": "search", "args": {"input": "q"}, "deps": []},
        ]
    )
    responses = [
        Response(text=dag_json, usage=Usage()),
        Response(text="async done", usage=Usage()),
    ]
    client = _make_async_client(responses)
    reg = _make_registry()
    agent = LLMCompilerAgent(client, reg)
    result = asyncio.run(agent.async_run("test"))

    assert result.answer == "async done"


def test_compiler_failed_task_status_preserved():
    """Failed task status is not overwritten to 'completed'."""

    def fail_tool(**kwargs: object) -> str:
        msg = "kaboom"
        raise RuntimeError(msg)

    dag_json = json.dumps(
        [
            {"id": "1", "tool": "fail", "args": {"input": "x"}, "deps": []},
        ]
    )
    responses = [
        Response(text=dag_json, usage=Usage()),
        Response(text="done", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    reg.register("fail", fail_tool, Tool(name="fail", description="", parameters={}))
    agent = LLMCompilerAgent(client, reg)
    result = agent.run("test")

    # Tool result should contain error
    assert "Error" in result.steps[-1].tool_results[0].content
    assert result.answer == "done"


def test_compiler_timeout():
    """LLMCompiler returns [timeout exceeded] when timeout is exceeded."""

    dag_json = json.dumps(
        [
            {"id": "1", "tool": "search", "args": {"input": "A"}, "deps": []},
            {"id": "2", "tool": "search", "args": {"input": "B"}, "deps": ["1"]},
            {"id": "3", "tool": "search", "args": {"input": "C"}, "deps": ["2"]},
        ]
    )

    def slow_chat(*args: object, **kwargs: object) -> Response:
        time.sleep(0.05)
        return Response(text=dag_json, usage=Usage())

    client = MagicMock()
    client.chat = MagicMock(side_effect=slow_chat)
    reg = _make_registry()
    config = AgentConfig(timeout=0.03)
    agent = LLMCompilerAgent(client, reg, config=config)
    result = agent.run("test")

    # Plan call takes 50ms, exceeds 30ms timeout.
    # Timeout fires at while-loop entry after plan completes.
    assert result.answer == "[timeout exceeded]"


def test_compiler_events_from_main_thread():
    """Events fire from the main thread, not from worker threads."""

    event_threads: list[str] = []
    main_thread = threading.current_thread().name

    def record_event(event: AgentEvent) -> None:
        event_threads.append(threading.current_thread().name)

    dag_json = json.dumps(
        [
            {"id": "1", "tool": "search", "args": {"input": "A"}, "deps": []},
            {"id": "2", "tool": "search", "args": {"input": "B"}, "deps": []},
        ]
    )
    responses = [
        Response(text=dag_json, usage=Usage()),
        Response(text="done", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = _make_registry()
    config = AgentConfig(on_event=record_event, parallel_tool_execution=True)
    agent = LLMCompilerAgent(client, reg, config=config)
    agent.run("test")

    # All events should fire from the main thread
    assert all(t == main_thread for t in event_threads)


def test_compiler_replan_triggered():
    """Re-planning executes a new DAG when joiner check says REPLAN."""
    dag1_json = json.dumps([{"id": "1", "tool": "search", "args": {"input": "A"}, "deps": []}])
    dag2_json = json.dumps([{"id": "2", "tool": "search", "args": {"input": "B"}, "deps": []}])
    responses = [
        Response(text=dag1_json, usage=Usage()),  # initial plan
        Response(text="REPLAN: need B", usage=Usage()),  # check → replan
        Response(text=dag2_json, usage=Usage()),  # re-plan DAG
        Response(text="Combined.", usage=Usage()),  # joiner
    ]
    client = _make_client(responses)
    reg = _make_registry()
    agent = LLMCompilerAgent(client, reg)
    result = agent.run("Search A and B", max_replans=1)

    assert result.answer == "Combined."
    assert client.chat.call_count == 4
    # Joiner should see both task results
    join_msg = client.chat.call_args_list[-1][0][0][0].content
    assert "found 'A'" in join_msg
    assert "found 'B'" in join_msg


def test_compiler_replan_finish_early():
    """Joiner check returning FINISH skips re-planning."""
    dag_json = json.dumps([{"id": "1", "tool": "search", "args": {"input": "q"}, "deps": []}])
    responses = [
        Response(text=dag_json, usage=Usage()),  # plan
        Response(text="FINISH", usage=Usage()),  # check → sufficient
        Response(text="Done.", usage=Usage()),  # joiner
    ]
    client = _make_client(responses)
    reg = _make_registry()
    agent = LLMCompilerAgent(client, reg)
    result = agent.run("test", max_replans=2)

    assert result.answer == "Done."
    # plan + check + joiner = 3 (no re-plan or second check)
    assert client.chat.call_count == 3


def test_compiler_replan_zero_default():
    """Default max_replans=0 means no replan check call happens."""
    dag_json = json.dumps([{"id": "1", "tool": "search", "args": {"input": "q"}, "deps": []}])
    responses = [
        Response(text=dag_json, usage=Usage()),
        Response(text="answer", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = _make_registry()
    agent = LLMCompilerAgent(client, reg)
    result = agent.run("test")

    assert result.answer == "answer"
    # plan + joiner = 2 (no check call)
    assert client.chat.call_count == 2
