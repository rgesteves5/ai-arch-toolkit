"""Tests for PhaseConfig — per-phase LLM/tools overrides."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._response import Response
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents._base import (
    AgentConfig,
    PhaseConfig,
    _resolve_llm,
    _resolve_tools,
)
from ai_arch_toolkit.toolkit.agents._lats import LATSAgent, LATSConfig
from ai_arch_toolkit.toolkit.agents._llm_compiler import LLMCompilerAgent, LLMCompilerConfig
from ai_arch_toolkit.toolkit.agents._plan_execute import PlanExecuteAgent, PlanExecuteConfig
from ai_arch_toolkit.toolkit.agents._reflexion import ReflexionAgent, ReflexionConfig
from ai_arch_toolkit.toolkit.agents._rewoo import ReWOOAgent, ReWOOConfig
from ai_arch_toolkit.toolkit.agents._self_discovery import SelfDiscoveryAgent, SelfDiscoveryConfig
from ai_arch_toolkit.toolkit.agents._tot import ToTAgent, ToTConfig
from tests.agents.conftest import make_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm(side_effect: list[Response] | None = None) -> AsyncMock:
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=side_effect)
    return llm


def _default_tools() -> ToolGroup:
    def lookup(query: str = "") -> str:
        """Look up information."""
        return f"info:{query}"

    return ToolGroup(lookup)


def _alt_tools() -> ToolGroup:
    def search(term: str = "") -> str:
        """Search for something."""
        return f"search:{term}"

    return ToolGroup(search)


# ---------------------------------------------------------------------------
# 1. PhaseConfig defaults — backward compat
# ---------------------------------------------------------------------------


def test_phase_config_defaults():
    pc = PhaseConfig()
    assert pc.llm is None
    assert pc.tools is None


def test_resolve_llm_default():
    default = _mock_llm()
    assert _resolve_llm(None, default) is default
    assert _resolve_llm(PhaseConfig(), default) is default


def test_resolve_llm_override():
    default = _mock_llm()
    override = _mock_llm()
    assert _resolve_llm(PhaseConfig(llm=override), default) is override


def test_resolve_tools_default():
    default = _default_tools()
    assert _resolve_tools(None, default) is default
    assert _resolve_tools(PhaseConfig(), default) is default


def test_resolve_tools_override():
    default = _default_tools()
    override = _alt_tools()
    assert _resolve_tools(PhaseConfig(tools=override), default) is override


# ---------------------------------------------------------------------------
# 2. PlanExecute: planner uses different LLM
# ---------------------------------------------------------------------------


async def test_plan_execute_planner_llm_override():
    planner_llm = _mock_llm([make_response(text="1. Lookup")])
    default_llm = _mock_llm(
        [
            make_response(text="Step result"),  # Execute
            make_response(text="Final"),  # Solve
        ]
    )
    tools = _default_tools()
    agent = PlanExecuteAgent(
        default_llm,
        tools,
        config=AgentConfig(),
        plan_execute=PlanExecuteConfig(planner=PhaseConfig(llm=planner_llm)),
    )
    result = await agent.run("test")

    assert result.stop_reason == "completed"
    # Planner LLM should have been called once (for planning)
    assert planner_llm.complete.call_count == 1
    # Default LLM used for execute + solve
    assert default_llm.complete.call_count == 2


# ---------------------------------------------------------------------------
# 3. LLMCompiler: executor uses different ToolGroup
# ---------------------------------------------------------------------------


async def test_llm_compiler_executor_tools_override():
    alt = _alt_tools()
    responses = [
        make_response(text="$1. Search for info [deps: none]"),  # Plan
        make_response(text="Found it"),  # Inner ReAct execute
        make_response(text="Final answer"),  # Join
    ]
    llm = _mock_llm(responses)
    default_tools = _default_tools()
    agent = LLMCompilerAgent(
        llm,
        default_tools,
        config=AgentConfig(),
        compiler=LLMCompilerConfig(executor=PhaseConfig(tools=alt)),
    )
    result = await agent.run("test")

    assert result.stop_reason == "completed"


# ---------------------------------------------------------------------------
# 4. Both LLM + tools overridden on same phase
# ---------------------------------------------------------------------------


async def test_plan_execute_executor_both_overrides():
    planner_resp = make_response(text="1. Do it")
    exec_resp = make_response(text="Exec result")
    solve_resp = make_response(text="Done")

    default_llm = _mock_llm([planner_resp, solve_resp])
    exec_llm = _mock_llm([exec_resp])
    exec_tools = _alt_tools()
    default_tools = _default_tools()

    agent = PlanExecuteAgent(
        default_llm,
        default_tools,
        config=AgentConfig(),
        plan_execute=PlanExecuteConfig(
            executor=PhaseConfig(llm=exec_llm, tools=exec_tools),
        ),
    )
    result = await agent.run("test")

    assert result.stop_reason == "completed"
    assert exec_llm.complete.call_count == 1
    assert default_llm.complete.call_count == 2  # planner + solver


# ---------------------------------------------------------------------------
# 5. Mixed: some phases overridden, some default
# ---------------------------------------------------------------------------


async def test_plan_execute_mixed_overrides():
    planner_llm = _mock_llm([make_response(text="1. Step")])
    solver_llm = _mock_llm([make_response(text="Solved")])
    default_llm = _mock_llm([make_response(text="Exec result")])

    agent = PlanExecuteAgent(
        default_llm,
        _default_tools(),
        config=AgentConfig(),
        plan_execute=PlanExecuteConfig(
            planner=PhaseConfig(llm=planner_llm),
            solver=PhaseConfig(llm=solver_llm),
            # executor uses default
        ),
    )
    result = await agent.run("test")

    assert result.stop_reason == "completed"
    assert planner_llm.complete.call_count == 1
    assert solver_llm.complete.call_count == 1
    assert default_llm.complete.call_count == 1  # executor only


# ---------------------------------------------------------------------------
# 6. Reflexion: executor + reflector phase overrides
# ---------------------------------------------------------------------------


async def test_reflexion_phase_overrides():
    exec_llm = _mock_llm(
        [
            make_response(text="Attempt answer"),  # First attempt
            make_response(text="Better answer"),  # Second attempt
        ]
    )
    reflect_llm = _mock_llm([make_response(text="Try harder")])
    default_llm = _mock_llm()

    call_count = {"eval": 0}

    def evaluator(task: str, answer: str) -> float:
        call_count["eval"] += 1
        return 0.5 if call_count["eval"] == 1 else 1.0

    agent = ReflexionAgent(
        default_llm,
        _default_tools(),
        config=AgentConfig(),
        reflexion=ReflexionConfig(
            evaluator=evaluator,
            executor=PhaseConfig(llm=exec_llm),
            reflector=PhaseConfig(llm=reflect_llm),
        ),
    )
    result = await agent.run("test")

    assert result.stop_reason == "completed"
    assert exec_llm.complete.call_count == 2  # Both attempts
    assert reflect_llm.complete.call_count == 1  # Reflection
    assert default_llm.complete.call_count == 0  # Not used


# ---------------------------------------------------------------------------
# 7. ReWOO: planner + solver phase overrides
# ---------------------------------------------------------------------------


async def test_rewoo_phase_overrides():
    planner_llm = _mock_llm(
        [
            make_response(text="#E1 = lookup[test]"),
        ]
    )
    solver_llm = _mock_llm([make_response(text="Solved")])
    default_llm = _mock_llm()

    agent = ReWOOAgent(
        default_llm,
        _default_tools(),
        config=AgentConfig(),
        rewoo=ReWOOConfig(
            planner=PhaseConfig(llm=planner_llm),
            solver=PhaseConfig(llm=solver_llm),
        ),
    )
    result = await agent.run("test")

    assert result.stop_reason == "completed"
    assert planner_llm.complete.call_count == 1
    assert solver_llm.complete.call_count == 1
    assert default_llm.complete.call_count == 0


# ---------------------------------------------------------------------------
# 8. SelfDiscovery: reasoning + solver phase overrides
# ---------------------------------------------------------------------------


async def test_self_discovery_phase_overrides():
    reasoning_llm = _mock_llm(
        [
            make_response(text="Critical thinking"),  # SELECT
            make_response(text="Adapted"),  # ADAPT
            make_response(text="Step-by-step plan"),  # OPERATIONALIZE
        ]
    )
    solver_llm = _mock_llm([make_response(text="Solved")])
    default_llm = _mock_llm()

    agent = SelfDiscoveryAgent(
        default_llm,
        _default_tools(),
        config=AgentConfig(),
        self_discovery=SelfDiscoveryConfig(
            reasoning=PhaseConfig(llm=reasoning_llm),
            solver=PhaseConfig(llm=solver_llm),
        ),
    )
    result = await agent.run("test")

    assert result.stop_reason == "completed"
    assert reasoning_llm.complete.call_count == 3
    assert solver_llm.complete.call_count == 1
    assert default_llm.complete.call_count == 0


# ---------------------------------------------------------------------------
# 9. ToT: generator + evaluator phase overrides
# ---------------------------------------------------------------------------


async def test_tot_phase_overrides():
    gen_llm = _mock_llm(
        [
            make_response(text="1. Candidate A"),  # Generate
        ]
    )
    eval_llm = _mock_llm(
        [
            make_response(text="0.95"),  # Evaluate (high score → early exit)
        ]
    )
    # Default LLM used for final answer after high-confidence detection
    default_llm = _mock_llm([make_response(text="Final answer")])

    agent = ToTAgent(
        default_llm,
        _default_tools(),
        config=AgentConfig(),
        tot=ToTConfig(
            n_candidates=1,
            generator=PhaseConfig(llm=gen_llm),
            evaluator=PhaseConfig(llm=eval_llm),
        ),
    )
    result = await agent.run("test")

    assert result.stop_reason == "completed"
    assert gen_llm.complete.call_count == 1
    assert eval_llm.complete.call_count == 1


# ---------------------------------------------------------------------------
# 10. LATS: rollout + evaluator_phase overrides
# ---------------------------------------------------------------------------


async def test_lats_phase_overrides():
    rollout_llm = _mock_llm([make_response(text="Attempt")])
    eval_llm = _mock_llm([make_response(text="0.95")])
    # Default for final answer
    default_llm = _mock_llm([make_response(text="Final")])

    agent = LATSAgent(
        default_llm,
        _default_tools(),
        config=AgentConfig(),
        lats=LATSConfig(
            rollout=PhaseConfig(llm=rollout_llm),
            evaluator=PhaseConfig(llm=eval_llm),
        ),
    )
    result = await agent.run("test")

    assert result.stop_reason == "completed"
    assert rollout_llm.complete.call_count == 1
    assert eval_llm.complete.call_count == 1


# ---------------------------------------------------------------------------
# 11. LLMCompiler: planner + joiner phase overrides
# ---------------------------------------------------------------------------


async def test_llm_compiler_phase_overrides():
    planner_llm = _mock_llm(
        [
            make_response(text="$1. Do task [deps: none]"),
        ]
    )
    joiner_llm = _mock_llm([make_response(text="Final answer")])
    # Default for inner execute
    default_llm = _mock_llm([make_response(text="Task result")])

    agent = LLMCompilerAgent(
        default_llm,
        _default_tools(),
        config=AgentConfig(),
        compiler=LLMCompilerConfig(
            planner=PhaseConfig(llm=planner_llm),
            joiner=PhaseConfig(llm=joiner_llm),
        ),
    )
    result = await agent.run("test")

    assert result.stop_reason == "completed"
    assert planner_llm.complete.call_count == 1
    assert joiner_llm.complete.call_count == 1
    assert default_llm.complete.call_count == 1  # executor


# ---------------------------------------------------------------------------
# 12. ToT: solver phase override
# ---------------------------------------------------------------------------


async def test_tot_solver_phase_override():
    gen_llm = _mock_llm([make_response(text="1. Candidate A")])
    eval_llm = _mock_llm([make_response(text="0.95")])
    solver_llm = _mock_llm([make_response(text="Solver final")])
    default_llm = _mock_llm()

    agent = ToTAgent(
        default_llm,
        _default_tools(),
        config=AgentConfig(),
        tot=ToTConfig(
            n_candidates=1,
            generator=PhaseConfig(llm=gen_llm),
            evaluator=PhaseConfig(llm=eval_llm),
            solver=PhaseConfig(llm=solver_llm),
        ),
    )
    result = await agent.run("test")

    assert result.stop_reason == "completed"
    assert solver_llm.complete.call_count == 1
    assert default_llm.complete.call_count == 0


# ---------------------------------------------------------------------------
# 13. LATS: solver + reflector phase overrides
# ---------------------------------------------------------------------------


async def test_lats_solver_phase_override():
    rollout_llm = _mock_llm([make_response(text="Attempt")])
    eval_llm = _mock_llm([make_response(text="0.95")])
    solver_llm = _mock_llm([make_response(text="Solver final")])
    default_llm = _mock_llm()

    agent = LATSAgent(
        default_llm,
        _default_tools(),
        config=AgentConfig(),
        lats=LATSConfig(
            rollout=PhaseConfig(llm=rollout_llm),
            evaluator=PhaseConfig(llm=eval_llm),
            solver=PhaseConfig(llm=solver_llm),
        ),
    )
    result = await agent.run("test")

    assert result.stop_reason == "completed"
    assert solver_llm.complete.call_count == 1
    assert default_llm.complete.call_count == 0


async def test_lats_reflector_phase_override():
    rollout_llm = _mock_llm(
        [
            make_response(text="Bad answer"),
            make_response(text="Better answer"),
        ]
    )
    eval_responses = [make_response(text="0.2"), make_response(text="0.95")]
    eval_llm = _mock_llm(eval_responses)
    reflector_llm = _mock_llm([make_response(text="Think harder")])
    solver_llm = _mock_llm([make_response(text="Final")])
    default_llm = _mock_llm()

    agent = LATSAgent(
        default_llm,
        _default_tools(),
        config=AgentConfig(),
        lats=LATSConfig(
            max_rollouts=5,
            rollout=PhaseConfig(llm=rollout_llm),
            evaluator=PhaseConfig(llm=eval_llm),
            solver=PhaseConfig(llm=solver_llm),
            reflector=PhaseConfig(llm=reflector_llm),
        ),
    )
    result = await agent.run("test")

    assert result.stop_reason == "completed"
    assert reflector_llm.complete.call_count == 1
    assert default_llm.complete.call_count == 0


# ---------------------------------------------------------------------------
# 14. PlanExecute: planner describes executor's tools (R1 fix)
# ---------------------------------------------------------------------------


async def test_planner_describes_executor_tools():
    alt = _alt_tools()  # has "search" tool
    default_tools = _default_tools()  # has "lookup" tool
    responses = [
        make_response(text="1. Search for info"),
        make_response(text="Found it"),
        make_response(text="Done"),
    ]
    llm = _mock_llm(responses)
    agent = PlanExecuteAgent(
        llm,
        default_tools,
        config=AgentConfig(),
        plan_execute=PlanExecuteConfig(executor=PhaseConfig(tools=alt)),
    )
    await agent.run("test")

    # Planner call system prompt should describe executor's tools (search), not default (lookup)
    plan_call = llm.complete.call_args_list[0]
    system = plan_call.kwargs.get("system", "")
    assert "search" in system
    assert "lookup" not in system


# ---------------------------------------------------------------------------
# 15. Import from top-level package
# ---------------------------------------------------------------------------


def test_top_level_import():
    from ai_arch_toolkit import PhaseConfig as PC

    assert PC is PhaseConfig
