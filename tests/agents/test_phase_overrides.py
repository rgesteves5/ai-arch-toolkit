"""Per-phase override routing and validation through ReasoningSpec builders.

Multi-phase strategies accept canonical runtime deps (``planner_llm``,
``executor_tools``, …) and canonical prompt knobs (``planner_system``, …).
These tests verify the right phase LLM receives the right call — and that the
default LLM receives none — plus builder-level validation of dep and knob
names, types, and values.
"""

from __future__ import annotations

from typing import Any

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents import Agent, ReasoningSpec, build_flow, get_strategy


def _make_response(text: str = "") -> Response:
    return Response(text=text, usage=Usage(input_tokens=10, output_tokens=5), cost=0.001)


class _RecordingProvider:
    """Real-LLM provider stand-in that records every call it receives."""

    def __init__(self, *texts: str) -> None:
        self._responses = [_make_response(text) for text in (texts or ("",))]
        self.calls: list[dict[str, Any]] = []

    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        self.calls.append({"messages": messages, "system": system, "kwargs": kwargs})
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


def _llm(*texts: str) -> LLM:
    llm = LLM("claude-sonnet-4-6", api_key="test")
    llm._provider = _RecordingProvider(*texts)  # type: ignore[assignment]
    return llm


def _calls(llm: LLM) -> list[dict[str, Any]]:
    return llm._provider.calls  # type: ignore[union-attr]


async def _run(spec: ReasoningSpec, llm: LLM, tools: ToolGroup | None = None, **deps: Any):
    agent = Agent(spec, llm, tools or ToolGroup(), deps=deps)
    return await agent.run("the task")


class TestPlanExecuteRouting:
    async def test_phase_llms_and_prompts(self) -> None:
        default = _llm("default")
        planner = _llm("1. Do the thing")
        executor = _llm("step done")
        solver = _llm("final answer")

        spec = ReasoningSpec(
            strategy="plan_execute",
            knobs={"planner_system": "PLAN IT", "solver_system": "SOLVE IT", "max_replans": 0},
        )
        result = await _run(
            spec, default, planner_llm=planner, executor_llm=executor, solver_llm=solver
        )

        assert result.text == "final answer"
        assert not _calls(default)
        assert len(_calls(planner)) == 1
        assert _calls(planner)[0]["system"] == "PLAN IT"
        assert len(_calls(executor)) == 1
        assert len(_calls(solver)) == 1
        assert _calls(solver)[0]["system"] == "SOLVE IT"

    async def test_planner_sees_executor_tools(self) -> None:
        def lookup(query: str) -> str:
            """Look up a fact."""
            return "fact"

        planner = _llm("1. Do the thing")
        spec = ReasoningSpec(strategy="plan_execute", knobs={"max_replans": 0})
        await _run(
            spec,
            _llm("default"),
            planner_llm=planner,
            executor_llm=_llm("done"),
            executor_tools=ToolGroup(lookup),
            solver_llm=_llm("final"),
        )

        planner_system = _calls(planner)[0]["system"]
        assert "Available tools:" in planner_system
        assert "lookup" in planner_system


class TestRewooRouting:
    async def test_phase_llms_and_prompts(self) -> None:
        default = _llm("default")
        planner = _llm("no tool calls needed")
        solver = _llm("final answer")

        spec = ReasoningSpec(
            strategy="rewoo",
            knobs={"planner_system": "PLAN IT", "solver_system": "SOLVE IT"},
        )
        result = await _run(spec, default, planner_llm=planner, solver_llm=solver)

        assert result.text == "final answer"
        assert not _calls(default)
        assert _calls(planner)[0]["system"] == "PLAN IT"
        assert _calls(solver)[0]["system"] == "SOLVE IT"


class TestReflexionRouting:
    async def test_executor_and_reflector(self) -> None:
        default = _llm("default")
        executor = _llm("attempt answer")
        reflector = _llm("try harder")
        scores = iter([0.0, 1.0])

        spec = ReasoningSpec(
            strategy="reflexion",
            knobs={"reflector_system": "REFLECT IT", "max_retries": 3},
        )
        result = await _run(
            spec,
            default,
            evaluator=lambda task, answer: next(scores),
            executor_llm=executor,
            reflector_llm=reflector,
        )

        assert result.text == "attempt answer"
        assert not _calls(default)
        assert len(_calls(executor)) == 2
        assert len(_calls(reflector)) == 1
        assert _calls(reflector)[0]["system"] == "REFLECT IT"


class TestSelfDiscoveryRouting:
    async def test_reasoning_and_solver(self) -> None:
        default = _llm("default")
        reasoning = _llm("selected", "adapted", "the plan")
        solver = _llm("final answer")

        spec = ReasoningSpec(
            strategy="self_discovery",
            knobs={
                "select_system": "SELECT",
                "adapt_system": "ADAPT",
                "plan_system": "PLAN",
                "solver_system": "SOLVE",
                "modules": ["Only Module: think hard."],
            },
        )
        result = await _run(spec, default, reasoning_llm=reasoning, solver_llm=solver)

        assert result.text == "final answer"
        assert not _calls(default)
        assert [c["system"] for c in _calls(reasoning)] == ["SELECT", "ADAPT", "PLAN"]
        assert "Only Module" in str(_calls(reasoning)[0]["messages"])
        assert len(_calls(solver)) == 1
        assert _calls(solver)[0]["system"].startswith("SOLVE")


class TestTotRouting:
    async def test_generator_evaluator_solver(self) -> None:
        default = _llm("default")
        generator = _llm("1. thought one")
        evaluator = _llm("0.95")
        solver = _llm("final answer")

        spec = ReasoningSpec(strategy="tot", knobs={"evaluator_system": "SCORE IT"})
        result = await _run(
            spec, default, generator_llm=generator, evaluator_llm=evaluator, solver_llm=solver
        )

        assert result.text == "final answer"
        assert not _calls(default)
        assert len(_calls(generator)) == 1
        assert _calls(evaluator)[0]["system"] == "SCORE IT"
        assert len(_calls(solver)) == 1


class TestLatsRouting:
    async def test_all_four_phases(self) -> None:
        default = _llm("default")
        rollout = _llm("attempt answer")
        evaluator = _llm("0.3")
        reflector = _llm("feedback")
        solver = _llm("final answer")

        spec = ReasoningSpec(
            strategy="lats",
            knobs={
                "max_rollouts": 1,
                "evaluator_system": "SCORE IT",
                "reflector_system": "REFLECT IT",
            },
        )
        result = await _run(
            spec,
            default,
            rollout_llm=rollout,
            evaluator_llm=evaluator,
            reflector_llm=reflector,
            solver_llm=solver,
        )

        assert result.text == "final answer"
        assert not _calls(default)
        assert len(_calls(rollout)) == 1
        assert _calls(evaluator)[0]["system"] == "SCORE IT"
        assert _calls(reflector)[0]["system"] == "REFLECT IT"
        assert len(_calls(solver)) == 1


class TestLlmCompilerRouting:
    async def test_planner_executor_joiner(self) -> None:
        default = _llm("default")
        planner = _llm("$1. Do the thing [deps: none]")
        executor = _llm("subtask done")
        joiner = _llm("final answer")

        spec = ReasoningSpec(
            strategy="llm_compiler",
            knobs={"planner_system": "PLAN IT", "joiner_system": "JOIN IT", "max_replans": 0},
        )
        result = await _run(
            spec, default, planner_llm=planner, executor_llm=executor, joiner_llm=joiner
        )

        assert result.text == "final answer"
        assert not _calls(default)
        assert _calls(planner)[0]["system"] == "PLAN IT"
        assert len(_calls(executor)) == 1
        assert _calls(joiner)[0]["system"] == "JOIN IT"


class TestGenerateReviewRouting:
    async def test_reviewer_phase(self) -> None:
        default = _llm("draft answer")
        reviewer = _llm("ACCEPT — looks good")

        spec = ReasoningSpec(strategy="generate_review", knobs={"reviewer_system": "REVIEW IT"})
        result = await _run(spec, default, reviewer_llm=reviewer)

        assert result.text == "draft answer"
        assert len(_calls(default)) == 1
        assert len(_calls(reviewer)) == 1
        assert _calls(reviewer)[0]["system"] == "REVIEW IT"

    async def test_legacy_alias_still_works(self) -> None:
        reviewer = _llm("ACCEPT")
        spec = ReasoningSpec(strategy="generate_review")
        await _run(spec, _llm("draft answer"), review_llm=reviewer)

        assert len(_calls(reviewer)) == 1

    async def test_both_aliases_rejected(self) -> None:
        spec = ReasoningSpec(strategy="generate_review")
        with pytest.raises(ValueError, match="legacy alias"):
            build_flow(
                spec, _llm(), ToolGroup(), deps={"reviewer_llm": _llm(), "review_llm": _llm()}
            )

    async def test_generator_phase_override(self) -> None:
        default = _llm("unused")
        generator = _llm("draft answer")
        reviewer = _llm("ACCEPT")

        spec = ReasoningSpec(strategy="generate_review")
        result = await _run(spec, default, generator_llm=generator, reviewer_llm=reviewer)

        assert result.text == "draft answer"
        assert not _calls(default)
        assert len(_calls(generator)) == 1

    async def test_llm_kwargs_reach_reviewer_with_precedence(self) -> None:
        default = _llm("draft")
        reviewer = _llm("ACCEPT")

        spec = ReasoningSpec(
            strategy="generate_review",
            llm_kwargs={"temperature": 0.2, "top_p": 0.9},
            knobs={"reviewer_kwargs": {"temperature": 0.0}},
        )
        await _run(spec, default, reviewer_llm=reviewer)

        kwargs = _calls(reviewer)[0]["kwargs"]
        assert kwargs.get("temperature") == 0.0  # reviewer_kwargs wins per key
        assert kwargs.get("top_p") == 0.9  # global llm_kwargs fill the rest


class TestToolsPlaceholder:
    async def test_custom_prompt_token_substituted_end_to_end(self) -> None:
        def lookup(query: str) -> str:
            """Look up a fact."""
            return "fact"

        planner = _llm("no tool calls needed")
        solver = _llm("final answer")
        spec = ReasoningSpec(
            strategy="rewoo",
            knobs={"planner_system": "Plan it.\nYou may reference:\n{tools}"},
        )
        await _run(spec, _llm(), ToolGroup(lookup), planner_llm=planner, solver_llm=solver)

        planner_system = _calls(planner)[0]["system"]
        assert "- lookup:" in planner_system
        assert "Look up a fact" in planner_system
        assert "{tools}" not in planner_system


class TestGlobalLlmKwargs:
    async def test_llm_kwargs_reach_every_phase(self) -> None:
        planner = _llm("1. Do the thing")
        executor = _llm("done")
        solver = _llm("final")

        spec = ReasoningSpec(
            strategy="plan_execute",
            llm_kwargs={"temperature": 0.2},
            knobs={"max_replans": 0},
        )
        await _run(spec, _llm(), planner_llm=planner, executor_llm=executor, solver_llm=solver)

        assert _calls(planner)[0]["kwargs"].get("temperature") == 0.2
        assert _calls(executor)[0]["kwargs"].get("temperature") == 0.2
        assert _calls(solver)[0]["kwargs"].get("temperature") == 0.2


class TestDepValidation:
    def test_unknown_dep_rejected(self) -> None:
        spec = ReasoningSpec(strategy="plan_execute")
        with pytest.raises(ValueError, match="unknown deps: plannr_llm"):
            build_flow(spec, _llm(), ToolGroup(), deps={"plannr_llm": _llm()})

    def test_wrong_type_dep_rejected(self) -> None:
        spec = ReasoningSpec(strategy="plan_execute")
        with pytest.raises(ValueError, match="invalid value for dep 'planner_llm'"):
            build_flow(spec, _llm(), ToolGroup(), deps={"planner_llm": "claude-haiku-4-5"})

    def test_tools_dep_type_checked(self) -> None:
        spec = ReasoningSpec(strategy="plan_execute")
        with pytest.raises(ValueError, match="executor_tools"):
            build_flow(spec, _llm(), ToolGroup(), deps={"executor_tools": ["not tools"]})

    def test_react_rejects_any_dep(self) -> None:
        spec = ReasoningSpec(strategy="react")
        with pytest.raises(ValueError, match="unknown deps"):
            build_flow(spec, _llm(), ToolGroup(), deps={"planner_llm": _llm()})

    def test_completion_rejects_any_dep(self) -> None:
        spec = ReasoningSpec(strategy="completion")
        with pytest.raises(ValueError, match="unknown deps"):
            build_flow(spec, _llm(), ToolGroup(), deps={"anything": object()})

    def test_evaluator_dep_still_accepted(self) -> None:
        spec = ReasoningSpec(strategy="reflexion")
        flow = build_flow(spec, _llm(), ToolGroup(), deps={"evaluator": lambda t, a: 1.0})
        assert flow is not None


class TestPhaseKnobValidation:
    def test_empty_prompt_rejected(self) -> None:
        spec = ReasoningSpec(strategy="plan_execute", knobs={"planner_system": ""})
        with pytest.raises(ValueError, match="invalid value"):
            build_flow(spec, _llm(), ToolGroup())

    def test_unknown_phase_knob_rejected(self) -> None:
        spec = ReasoningSpec(strategy="rewoo", knobs={"planner_sytem": "typo"})
        with pytest.raises(ValueError, match="unknown knobs"):
            build_flow(spec, _llm(), ToolGroup())

    def test_modules_must_be_nonempty_strings(self) -> None:
        spec = ReasoningSpec(strategy="self_discovery", knobs={"modules": ["ok", ""]})
        with pytest.raises(ValueError, match="modules"):
            build_flow(spec, _llm(), ToolGroup())

    def test_exploration_weight_must_be_positive(self) -> None:
        spec = ReasoningSpec(strategy="lats", knobs={"exploration_weight": -1})
        with pytest.raises(ValueError, match="exploration_weight"):
            build_flow(spec, _llm(), ToolGroup())


class TestPhasesMetadata:
    def test_builtin_phase_names(self) -> None:
        assert get_strategy("plan_execute").phases == {"planner", "executor", "solver"}
        assert get_strategy("lats").phases == {"rollout", "evaluator", "solver", "reflector"}
        assert get_strategy("react").phases == frozenset()
