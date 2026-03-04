"""Tests for ToTAgent — Tree of Thoughts."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._response import Response
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents._base import AgentConfig, AgentEvent, AgentResult
from ai_arch_toolkit.toolkit.agents._tot import ToTAgent, ToTConfig
from tests.agents.conftest import make_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    llm_side_effect: list[Response],
    *,
    config: AgentConfig | None = None,
    tot: ToTConfig | None = None,
) -> ToTAgent:
    """Build a ToTAgent with mocked LLM."""

    def dummy_tool(x: str = "") -> str:
        """A test tool."""
        return f"result:{x}"

    tools = ToolGroup(dummy_tool)
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=llm_side_effect)
    return ToTAgent(
        llm,
        tools,
        config=config or AgentConfig(),
        tot=tot,
    )


# ---------------------------------------------------------------------------
# 1. Finds solution at depth 0 (high score immediately)
# ---------------------------------------------------------------------------


async def test_finds_solution_immediately():
    responses = [
        # Generate candidates
        make_response(text="1. Great thought\n2. OK thought\n3. Bad thought"),
        # Evaluate candidate 1 → high score
        make_response(text="0.95"),
        # Evaluate candidate 2
        make_response(text="0.3"),
        # Evaluate candidate 3
        make_response(text="0.2"),
        # Final answer (triggered by 0.95 >= 0.9)
        make_response(text="The answer is 42"),
    ]
    agent = _make_agent(responses, tot=ToTConfig(n_candidates=3))
    result = await agent.run("What is 6*7?")

    assert result.stop_reason == "completed"
    assert result.answer == "The answer is 42"


# ---------------------------------------------------------------------------
# 2. DFS explores deeper before wider
# ---------------------------------------------------------------------------


async def test_dfs_explores_deeper():
    # DFS: sorted by score desc → [A(0.5), B(0.4)] added left-to-right.
    # DFS pops from right → picks B (last appended) at depth 1.
    # But since sorted desc, A is appended first, B second. DFS pops B.
    # Wait — sorted desc: [(0.5, A), (0.4, B)]. Iterate in that order.
    # frontier.append(A, depth=1), then frontier.append(B, depth=1).
    # DFS pops B (right side).
    responses = [
        # Depth 0: generate 2 candidates
        make_response(text="1. A\n2. B"),
        make_response(text="0.5"),  # eval A
        make_response(text="0.4"),  # eval B
        # Depth 1: DFS pops B (last added). Generate from B's state.
        make_response(text="1. Deep from B"),
        make_response(text="0.95"),  # high score → done
        make_response(text="Final deep answer"),
    ]
    agent = _make_agent(
        responses,
        config=AgentConfig(max_iterations=10),
        tot=ToTConfig(n_candidates=2, strategy="dfs"),
    )
    result = await agent.run("Solve this")

    assert result.stop_reason == "completed"

    # Verify depth 1 generation was called with B's state (DFS popped last-added)
    depth1_call = agent.llm.complete.call_args_list[3]  # 4th call = generate at depth 1
    depth1_prompt = depth1_call.args[0][0]["content"]
    assert "B" in depth1_prompt  # DFS explored B before A


# ---------------------------------------------------------------------------
# 3. BFS explores level by level
# ---------------------------------------------------------------------------


async def test_bfs_explores_level_by_level():
    # BFS: sorted desc → [A(0.5), B(0.4)] appended left-to-right.
    # BFS pops from left → picks A (first appended) at depth 1.
    responses = [
        # Level 0: root
        make_response(text="1. A\n2. B"),
        make_response(text="0.5"),  # eval A
        make_response(text="0.4"),  # eval B
        # Level 1: BFS pops A (first added). Generate from A's state.
        make_response(text="1. AA"),
        make_response(text="0.95"),  # high score
        make_response(text="BFS answer"),
    ]
    agent = _make_agent(
        responses,
        config=AgentConfig(max_iterations=10),
        tot=ToTConfig(n_candidates=2, strategy="bfs"),
    )
    result = await agent.run("Solve this")

    assert result.stop_reason == "completed"

    # Verify depth 1 generation was called with A's state (BFS popped first-added)
    depth1_call = agent.llm.complete.call_args_list[3]  # 4th call = generate at depth 1
    depth1_prompt = depth1_call.args[0][0]["content"]
    assert "A" in depth1_prompt  # BFS explored A before B


# ---------------------------------------------------------------------------
# 4. Max iterations exhausted
# ---------------------------------------------------------------------------


async def test_max_iterations_exhausted():
    # All evaluations return low scores, exhaust max_iterations
    responses = [
        make_response(text="1. Thought"),
        make_response(text="0.3"),  # low score
    ]
    agent = _make_agent(
        responses,
        config=AgentConfig(max_iterations=1),
        tot=ToTConfig(n_candidates=1, max_depth=5),
    )
    result = await agent.run("Hard problem")

    assert result.stop_reason == "max_iterations"


# ---------------------------------------------------------------------------
# 5. Event sequence (step_start/step_end per node)
# ---------------------------------------------------------------------------


async def test_event_sequence():
    responses = [
        make_response(text="1. Thought A\n2. Thought B"),
        make_response(text="0.95"),  # A gets high score
        make_response(text="0.2"),
        make_response(text="Final"),
    ]
    events: list[AgentEvent] = []
    config = AgentConfig(on_event=events.append, max_iterations=10)
    agent = _make_agent(
        responses,
        config=config,
        tot=ToTConfig(n_candidates=2),
    )
    await agent.run("Test events")

    types = [e.type for e in events]
    # Should have step_start/step_end pairs
    assert types.count("step_start") >= 1
    assert types.count("step_end") >= 1
    # Step numbers should increase
    step_starts = [e.step for e in events if e.type == "step_start"]
    assert step_starts == sorted(step_starts)


# ---------------------------------------------------------------------------
# 6. run_sync works
# ---------------------------------------------------------------------------


def test_run_sync():
    responses = [
        make_response(text="1. Quick thought"),
        make_response(text="0.95"),
        make_response(text="Sync answer"),
    ]
    agent = _make_agent(
        responses,
        tot=ToTConfig(n_candidates=1),
    )
    result = agent.run_sync("Sync test")

    assert isinstance(result, AgentResult)
    assert result.stop_reason == "completed"


# ---------------------------------------------------------------------------
# 7. Custom evaluator_system forwarded
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7b. Timeout stops search
# ---------------------------------------------------------------------------


async def test_timeout_stops_search(monkeypatch):
    import ai_arch_toolkit.toolkit.agents._tot as tot_mod

    _counter = {"n": 0}

    def fake_monotonic():
        _counter["n"] += 1
        return _counter["n"] * 100.0

    monkeypatch.setattr(tot_mod.time, "monotonic", fake_monotonic)

    responses = [
        make_response(text="1. Thought"),
        make_response(text="0.3"),
    ]
    agent = _make_agent(
        responses,
        config=AgentConfig(timeout=0.001, max_iterations=10),
        tot=ToTConfig(n_candidates=1),
    )
    result = await agent.run("Test timeout")

    assert result.stop_reason == "timeout"


# ---------------------------------------------------------------------------
# 8. Custom evaluator_system forwarded
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7d. Max depth produces final answer
# ---------------------------------------------------------------------------


async def test_max_depth_produces_final_answer():
    """When depth >= max_depth, the agent should produce a final answer directly."""
    responses = [
        # Depth 0: generate candidates
        make_response(text="1. Thought A"),
        # Evaluate candidate → low score (not high enough to exit)
        make_response(text="0.5"),
        # Depth 1 (== max_depth): final answer directly from state
        make_response(text="Max depth answer"),
    ]
    agent = _make_agent(
        responses,
        config=AgentConfig(max_iterations=10),
        tot=ToTConfig(n_candidates=1, max_depth=1),
    )
    result = await agent.run("Test max depth")

    assert result.stop_reason == "completed"
    assert result.answer == "Max depth answer"


# ---------------------------------------------------------------------------
# 7e. Parse helpers edge cases
# ---------------------------------------------------------------------------


def test_parse_numbered_items_fallback():
    """When no numbered items found, falls back to splitlines."""
    from ai_arch_toolkit.toolkit.agents._tot import _parse_numbered_items

    # No numbered items → splitlines
    result = _parse_numbered_items("line one\nline two\nline three")
    assert result == ["line one", "line two", "line three"]

    # Empty string
    result = _parse_numbered_items("")
    assert result == []


def test_parse_score_edge_cases():
    """Score parsing boundary and edge cases."""
    from ai_arch_toolkit.toolkit.agents._tot import _parse_score

    assert _parse_score("0.5") == 0.5
    assert _parse_score("no score here") == 0.0
    assert _parse_score("") == 0.0
    assert _parse_score("1.5") == 1.0  # clamped to 1.0
    assert _parse_score("0.0") == 0.0
    assert _parse_score("The score is 0.75 out of 1.0") == 0.75


# ---------------------------------------------------------------------------
# 7f. Empty frontier (all at max depth)
# ---------------------------------------------------------------------------


async def test_empty_frontier_exhaustion():
    """Frontier becomes empty when all candidates expand to max depth."""
    responses = [
        # Depth 0: generate 1 candidate
        make_response(text="1. Only thought"),
        make_response(text="0.5"),
        # Depth 1 == max_depth: final answer
        make_response(text="Answer from depth"),
    ]
    agent = _make_agent(
        responses,
        config=AgentConfig(max_iterations=10),
        tot=ToTConfig(n_candidates=1, max_depth=1),
    )
    result = await agent.run("Test frontier")
    assert result.stop_reason == "completed"


# ---------------------------------------------------------------------------
# 8. Custom evaluator_system forwarded
# ---------------------------------------------------------------------------


async def test_custom_evaluator_system():
    custom_sys = "Rate creativity 0-1."
    responses = [
        make_response(text="1. Creative idea"),
        make_response(text="0.95"),
        make_response(text="Final creative answer"),
    ]
    agent = _make_agent(
        responses,
        tot=ToTConfig(n_candidates=1, evaluator_system=custom_sys),
    )
    result = await agent.run("Be creative")

    assert result.stop_reason == "completed"
    # The evaluator call should use the custom system prompt
    eval_call = agent.llm.complete.call_args_list[1]
    assert eval_call.kwargs.get("system") == custom_sys
