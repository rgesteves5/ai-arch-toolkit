"""Tests for the Layer 1 configurable-agent compiler (spec -> Flow -> run)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import OutputSchema, Response, ToolCall, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._step import Result
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.core._trace import StepTrace, Trace
from ai_arch_toolkit.toolkit.agents import (
    Agent,
    FlowStrategy,
    ReasoningSpec,
    build_flow,
    get_strategy,
    register_strategy,
    strategy_names,
)
from ai_arch_toolkit.toolkit.agents.flows._react import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.flow._flow import Flow, FlowResult

_BUILTINS = {
    "react",
    "completion",
    "plan_execute",
    "rewoo",
    "reflexion",
    "generate_review",
    "self_discovery",
    "llm_compiler",
    "tot",
    "lats",
}


def _make_response(
    text: str = "", tool_calls: tuple[ToolCall, ...] = (), cost: float = 0.001
) -> Response:
    return Response(
        text=text,
        tool_calls=tool_calls,
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=cost,
    )


class _FakeProvider:
    """A real LLM's provider stand-in, so LLM.complete runs its metering charge site."""

    def __init__(self, *responses: Response) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


def _metered_llm(*responses: Response) -> LLM:
    llm = LLM("claude-sonnet-4-6", api_key="test")
    llm._provider = _FakeProvider(*responses)  # type: ignore[assignment]
    return llm


class TestReasoningSpec:
    def test_from_mapping_parses_fields(self) -> None:
        spec = ReasoningSpec.from_mapping(
            {
                "strategy": "tot",
                "system": "sys",
                "max_iterations": 4,
                "knobs": {"n_candidates": 5},
                "llm_kwargs": {"temperature": 0.2},
            }
        )
        assert spec.strategy == "tot"
        assert spec.system == "sys"
        assert spec.max_iterations == 4
        assert spec.knobs["n_candidates"] == 5
        assert spec.llm_kwargs["temperature"] == 0.2

    def test_from_mapping_defaults(self) -> None:
        spec = ReasoningSpec.from_mapping({})
        assert spec.strategy == "react"
        assert spec.max_iterations == 10
        assert spec.output_schema is None

    def test_output_schema_coercion_from_dict(self) -> None:
        spec = ReasoningSpec.from_mapping(
            {"output_schema": {"name": "out", "schema": {"type": "object"}}}
        )
        assert spec.output_schema is not None
        assert spec.output_schema.name == "out"

    def test_output_schema_accepts_model_class(self) -> None:
        class StructuredResult:
            pass

        spec = ReasoningSpec.from_mapping({"output_schema": StructuredResult})

        assert spec.output_schema is StructuredResult


class TestRegistry:
    def test_builtins_are_registered(self) -> None:
        assert set(strategy_names()) >= _BUILTINS

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown strategy"):
            get_strategy("nope")

    def test_register_custom_strategy(self) -> None:
        register_strategy(
            "custom_noop",
            FlowStrategy(
                lambda ctx: react_flow(ctx.llm, ctx.tools, max_iterations=1),
                react_initial_state,
            ),
        )
        assert "custom_noop" in strategy_names()
        assert get_strategy("custom_noop").supports_output_schema is False


class TestBuildFlow:
    def test_builds_a_flow_for_every_builtin(self) -> None:
        llm = AsyncMock()
        tools = ToolGroup()
        for name in _BUILTINS:
            flow = build_flow(ReasoningSpec(strategy=name), llm, tools)
            assert isinstance(flow, Flow)

    def test_output_schema_allowed_on_react(self) -> None:
        schema = OutputSchema(name="out", schema={"type": "object"})
        flow = build_flow(
            ReasoningSpec(strategy="react", output_schema=schema), AsyncMock(), ToolGroup()
        )
        assert isinstance(flow, Flow)

    def test_model_class_output_schema_allowed_on_completion(self) -> None:
        class StructuredResult:
            pass

        flow = build_flow(
            ReasoningSpec(strategy="completion", output_schema=StructuredResult),
            AsyncMock(),
            ToolGroup(),
        )

        assert isinstance(flow, Flow)

    def test_output_schema_allowed_on_generate_review(self) -> None:
        schema = OutputSchema(name="out", schema={"type": "object"})
        flow = build_flow(
            ReasoningSpec(strategy="generate_review", output_schema=schema),
            AsyncMock(),
            ToolGroup(),
        )

        assert isinstance(flow, Flow)
        assert get_strategy("generate_review").supports_output_schema is True

    def test_output_schema_rejected_on_unsupported_strategy(self) -> None:
        schema = OutputSchema(name="out", schema={"type": "object"})
        with pytest.raises(ValueError, match="does not support output_schema"):
            build_flow(
                ReasoningSpec(strategy="plan_execute", output_schema=schema),
                AsyncMock(),
                ToolGroup(),
            )

    @pytest.mark.parametrize("strategy", sorted(_BUILTINS))
    def test_unknown_strategy_knob_is_rejected(self, strategy: str) -> None:
        with pytest.raises(ValueError, match="unknown knobs"):
            build_flow(
                ReasoningSpec(strategy=strategy, knobs={"misspelled_knob": 1}),
                AsyncMock(),
                ToolGroup(),
            )

    @pytest.mark.parametrize(
        ("strategy", "knobs"),
        [
            ("react", {"parallel_tool_calls": "yes"}),
            ("plan_execute", {"max_replans": -1}),
            ("reflexion", {"threshold": 1.5}),
            ("generate_review", {"max_cycles": 0}),
            ("llm_compiler", {"max_replans": -1}),
            ("tot", {"search_strategy": "random"}),
            ("tot", {"search_strategy": ["dfs"]}),
            ("lats", {"n_candidates": 0}),
        ],
    )
    def test_invalid_strategy_knob_value_is_rejected(
        self, strategy: str, knobs: dict[str, object]
    ) -> None:
        with pytest.raises(ValueError, match="invalid value"):
            build_flow(ReasoningSpec(strategy=strategy, knobs=knobs), AsyncMock(), ToolGroup())

    @pytest.mark.parametrize(
        ("strategy", "knob"),
        [("reflexion", "evaluator"), ("lats", "evaluator_fn")],
    )
    def test_runtime_evaluators_are_not_serializable_knobs(self, strategy: str, knob: str) -> None:
        with pytest.raises(ValueError, match="unknown knobs"):
            build_flow(
                ReasoningSpec(strategy=strategy, knobs={knob: lambda *_: 1.0}),
                AsyncMock(),
                ToolGroup(),
            )

    @pytest.mark.parametrize(
        ("strategy", "dependency"),
        [("reflexion", "evaluator"), ("lats", "evaluator_fn")],
    )
    def test_runtime_evaluators_are_accepted_through_deps(
        self, strategy: str, dependency: str
    ) -> None:
        flow = build_flow(
            ReasoningSpec(strategy=strategy),
            AsyncMock(),
            ToolGroup(),
            deps={dependency: lambda *_: 1.0},
        )
        assert isinstance(flow, Flow)

    @pytest.mark.parametrize(
        ("strategy", "dependency"),
        [("reflexion", "evaluator"), ("lats", "evaluator_fn")],
    )
    def test_runtime_evaluators_must_be_callable(self, strategy: str, dependency: str) -> None:
        with pytest.raises(ValueError, match="must be callable"):
            build_flow(
                ReasoningSpec(strategy=strategy),
                AsyncMock(),
                ToolGroup(),
                deps={dependency: "not callable"},
            )


class TestAgentRun:
    async def test_completion_agent_runs(self) -> None:
        # Real LLM + fake provider so the metering charge site runs (usage is meter-derived).
        llm = _metered_llm(_make_response(text="Hello!"))
        agent = Agent(ReasoningSpec(strategy="completion"), llm)

        result = await agent.run("Hi")

        assert result.text == "Hello!"
        assert result.response is not None
        assert result.usage.input_tokens == 10
        assert result.errors == ()

    async def test_structured_response_without_text_is_preserved(self) -> None:
        """A parsed response is valid even when its string payload is empty."""
        parsed = {"answer": 42}
        response = Response(parsed=parsed, usage=Usage(input_tokens=3, output_tokens=2))
        agent = Agent(ReasoningSpec(strategy="completion"), _metered_llm(response))

        result = await agent.run("Return JSON")

        assert result.response is not None
        assert result.response.parsed is parsed
        assert result.text == ""

    async def test_react_agent_runs(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_response(text="Done"))
        agent = Agent(ReasoningSpec(strategy="react", max_iterations=3), llm, ToolGroup())

        result = await agent.run("Question")

        assert result.text == "Done"

    def test_run_sync(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_response(text="Sync!"))
        agent = Agent(ReasoningSpec(strategy="completion"), llm)

        result = agent.run_sync("Hi")

        assert result.text == "Sync!"

    async def test_errors_retain_earlier_failure_for_repeated_step_name(self) -> None:
        flow = AsyncMock()
        flow.run.return_value = FlowResult(
            state=State(),
            trace=Trace(
                flow_name="cyclic",
                steps=(
                    StepTrace(name="turn", error="first turn failed"),
                    StepTrace(name="turn"),
                ),
            ),
            results={"turn": Result(value="eventual success")},
        )
        agent = Agent.from_flow(flow)

        result = await agent.run("Question")

        assert result.errors == ("first turn failed",)


class TestEscapeHatch:
    async def test_from_flow_runs_arbitrary_flow(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_response(text="Custom"))
        flow = react_flow(llm, ToolGroup(), max_iterations=2)

        agent = Agent.from_flow(flow)
        result = await agent.run("Question")

        assert result.text == "Custom"

    async def test_custom_strategy_via_agent(self) -> None:
        register_strategy(
            "echo_custom",
            FlowStrategy(
                lambda ctx: react_flow(ctx.llm, ctx.tools, max_iterations=1),
                react_initial_state,
            ),
        )
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_response(text="Echo"))
        agent = Agent(ReasoningSpec(strategy="echo_custom"), llm, ToolGroup())

        result = await agent.run("Question")

        assert result.text == "Echo"
