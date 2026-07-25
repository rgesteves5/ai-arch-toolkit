"""Strategy builders — turn a ReasoningSpec into a runnable Flow.

A *strategy* is a named recipe for how an agent reasons. Each built-in strategy
adapts an existing Flow factory; consumers register their own with
``register_strategy``. A ``BuildContext`` separates serializable config
(``spec.knobs``) from runtime dependencies (``deps`` — an evaluator callable, a
second LLM, a memory store) that cannot live in a config file.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable

from ai_arch_toolkit.core._content import Content, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._metering._admission import AdmissionDenied
from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._state import StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents._spec import ReasoningSpec
from ai_arch_toolkit.toolkit.agents.flows import (
    generate_review_flow,
    generate_review_initial_state,
    lats_flow,
    lats_initial_state,
    llm_compiler_flow,
    llm_compiler_initial_state,
    plan_execute_flow,
    plan_execute_initial_state,
    react_flow,
    react_initial_state,
    reflexion_flow,
    reflexion_initial_state,
    rewoo_flow,
    rewoo_initial_state,
    self_discovery_flow,
    self_discovery_initial_state,
    tot_flow,
    tot_initial_state,
)
from ai_arch_toolkit.toolkit.flow._flow import Flow

__all__ = [
    "BuildContext",
    "FlowStrategy",
    "StrategyBuilder",
    "get_strategy",
    "register_strategy",
    "strategy_names",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class BuildContext:
    """Everything a strategy needs to build its Flow.

    ``deps`` carries runtime objects a strategy may require but that cannot be
    serialized — an evaluator callable, a second LLM, a memory store.
    """

    spec: ReasoningSpec
    llm: LLM
    tools: ToolGroup
    deps: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class StrategyBuilder(Protocol):
    """A named recipe that turns a BuildContext into a runnable Flow."""

    @property
    def supports_output_schema(self) -> bool: ...

    def build(self, ctx: BuildContext) -> Flow: ...

    def init_state(self, task: Content) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class FlowStrategy:
    """Adapts a Flow factory plus an initial-state function into a builder."""

    builder: Callable[[BuildContext], Flow]
    initializer: Callable[[Content], dict[str, Any]]
    supports_output_schema: bool = False
    allowed_knobs: frozenset[str] | None = None
    knob_validators: Mapping[str, Callable[[Any], bool]] = field(default_factory=dict)

    def build(self, ctx: BuildContext) -> Flow:
        if self.allowed_knobs is not None:
            unknown = sorted(set(ctx.spec.knobs) - self.allowed_knobs)
            if unknown:
                raise ValueError(
                    f"strategy {ctx.spec.strategy!r} received unknown knobs: {', '.join(unknown)}"
                )
        for name, validator in self.knob_validators.items():
            if name in ctx.spec.knobs and not validator(ctx.spec.knobs[name]):
                raise ValueError(
                    f"strategy {ctx.spec.strategy!r} received invalid value "
                    f"for knob {name!r}: {ctx.spec.knobs[name]!r}"
                )
        return self.builder(ctx)

    def init_state(self, task: Content) -> dict[str, Any]:
        return self.initializer(task)


_REGISTRY: dict[str, StrategyBuilder] = {}


def register_strategy(name: str, builder: StrategyBuilder) -> None:
    """Register a strategy under a stable name (overwrites an existing one)."""
    if not name:
        raise ValueError("strategy name must be non-empty")
    _REGISTRY[name] = builder


def get_strategy(name: str) -> StrategyBuilder:
    """Look up a registered strategy, raising a clear error if unknown."""
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(strategy_names())
        raise ValueError(f"unknown strategy {name!r}; registered: {known}") from None


def strategy_names() -> tuple[str, ...]:
    """Return the registered strategy names, sorted."""
    return tuple(sorted(_REGISTRY))


def _default_evaluator(_task: str, answer: str) -> float:
    return 1.0 if answer.strip() else 0.0


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_probability(value: Any) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= float(value) <= 1
    )


def _is_search_strategy(value: Any) -> bool:
    return isinstance(value, str) and value in {"bfs", "dfs"}


def _build_react(ctx: BuildContext) -> Flow:
    s = ctx.spec
    llm_kwargs = dict(s.llm_kwargs)
    if s.output_schema is not None:
        llm_kwargs.setdefault("output_schema", s.output_schema)
    return react_flow(
        ctx.llm,
        ctx.tools,
        system=s.system,
        max_iterations=s.max_iterations,
        parallel_tool_calls=s.knobs.get("parallel_tool_calls", True),
        timeout=s.timeout,
        policy=s.policy,
        llm_kwargs=llm_kwargs or None,
        final_answer_hint=s.knobs.get("final_answer_hint", True),
        strip_tools_on_final=s.knobs.get("strip_tools_on_final", False),
        show_turn_counter=s.knobs.get("show_turn_counter", False),
    )


def _build_completion(ctx: BuildContext) -> Flow:
    s = ctx.spec
    llm = ctx.llm
    system = s.system or None
    llm_kwargs = dict(s.llm_kwargs)
    if s.output_schema is not None:
        llm_kwargs.setdefault("output_schema", s.output_schema)

    async def _complete(snap: StateSnapshot) -> Result:
        messages = snap.require("messages")
        try:
            response = await llm.complete(messages, system=system, **llm_kwargs)
        except AdmissionDenied:
            raise  # budget denial is terminal — the flow executor converts it to budget_exceeded
        except Exception as exc:
            return Result(error=str(exc))
        return Result(
            value=response,
            artifacts={"response": response, "answer": response.text},
        )

    flow_policy = s.policy
    if s.timeout is not None and flow_policy is None:
        flow_policy = Policy(timeout=s.timeout)
    return Flow(Step(name="complete", fn=_complete), name="completion", policy=flow_policy)


def _completion_initial_state(task: Content) -> dict[str, Any]:
    return {"messages": [user(task)]}


def _build_plan_execute(ctx: BuildContext) -> Flow:
    s = ctx.spec
    return plan_execute_flow(
        ctx.llm,
        ctx.tools,
        system=s.system,
        max_replans=s.knobs.get("max_replans", 1),
        max_iterations_per_step=s.knobs.get("max_iterations_per_step", s.max_iterations),
        timeout=s.timeout,
        policy=s.policy,
    )


def _build_rewoo(ctx: BuildContext) -> Flow:
    s = ctx.spec
    return rewoo_flow(ctx.llm, ctx.tools, system=s.system, timeout=s.timeout, policy=s.policy)


def _build_reflexion(ctx: BuildContext) -> Flow:
    s = ctx.spec
    configured_evaluator = ctx.deps.get("evaluator", _default_evaluator)
    if not callable(configured_evaluator):
        raise ValueError("strategy 'reflexion' dependency 'evaluator' must be callable")
    evaluator = cast(Callable[[str, str], float], configured_evaluator)
    return reflexion_flow(
        ctx.llm,
        ctx.tools,
        evaluator=evaluator,
        threshold=s.knobs.get("threshold", 0.7),
        max_retries=s.knobs.get("max_retries", 3),
        system=s.system,
        max_iterations=s.max_iterations,
        timeout=s.timeout,
        policy=s.policy,
    )


def _build_generate_review(ctx: BuildContext) -> Flow:
    s = ctx.spec
    review_llm = ctx.deps.get("review_llm", ctx.llm)
    review_tools = ctx.deps.get("review_tools", ctx.tools)
    return generate_review_flow(
        gen_llm=ctx.llm,
        review_llm=review_llm,
        gen_tools=ctx.tools,
        review_tools=review_tools,
        gen_system=s.system,
        gen_kwargs=dict(s.llm_kwargs) or None,
        max_cycles=s.knobs.get("max_cycles", 3),
        max_gen_iterations=s.max_iterations,
        max_review_iterations=s.knobs.get("max_review_iterations", 5),
        timeout=s.timeout,
        policy=s.policy,
    )


def _build_self_discovery(ctx: BuildContext) -> Flow:
    s = ctx.spec
    return self_discovery_flow(
        ctx.llm,
        ctx.tools,
        system=s.system,
        max_react_iterations=s.max_iterations,
        timeout=s.timeout,
        policy=s.policy,
    )


def _build_llm_compiler(ctx: BuildContext) -> Flow:
    s = ctx.spec
    return llm_compiler_flow(
        ctx.llm,
        ctx.tools,
        system=s.system,
        max_replans=s.knobs.get("max_replans", 2),
        max_react_iterations=s.max_iterations,
        timeout=s.timeout,
        policy=s.policy,
    )


def _build_tot(ctx: BuildContext) -> Flow:
    s = ctx.spec
    return tot_flow(
        ctx.llm,
        ctx.tools,
        system=s.system,
        n_candidates=s.knobs.get("n_candidates", 3),
        max_depth=s.knobs.get("max_depth", 3),
        max_iterations=s.max_iterations,
        strategy=s.knobs.get("search_strategy", "dfs"),
        timeout=s.timeout,
        policy=s.policy,
    )


def _build_lats(ctx: BuildContext) -> Flow:
    s = ctx.spec
    configured_evaluator = ctx.deps.get("evaluator_fn")
    if configured_evaluator is not None and not callable(configured_evaluator):
        raise ValueError("strategy 'lats' dependency 'evaluator_fn' must be callable")
    evaluator_fn = cast(Callable[[str, str], float] | None, configured_evaluator)
    return lats_flow(
        ctx.llm,
        ctx.tools,
        system=s.system,
        n_candidates=s.knobs.get("n_candidates", 5),
        max_rollouts=s.knobs.get("max_rollouts", s.max_iterations),
        max_react_iterations=s.max_iterations,
        evaluator_fn=evaluator_fn,
        timeout=s.timeout,
        policy=s.policy,
    )


register_strategy(
    "react",
    FlowStrategy(
        _build_react,
        react_initial_state,
        supports_output_schema=True,
        allowed_knobs=frozenset(
            {
                "final_answer_hint",
                "parallel_tool_calls",
                "show_turn_counter",
                "strip_tools_on_final",
            }
        ),
        knob_validators={
            "final_answer_hint": _is_bool,
            "parallel_tool_calls": _is_bool,
            "show_turn_counter": _is_bool,
            "strip_tools_on_final": _is_bool,
        },
    ),
)
register_strategy(
    "completion",
    FlowStrategy(
        _build_completion,
        _completion_initial_state,
        supports_output_schema=True,
        allowed_knobs=frozenset(),
    ),
)
register_strategy(
    "plan_execute",
    FlowStrategy(
        _build_plan_execute,
        plan_execute_initial_state,
        allowed_knobs=frozenset({"max_iterations_per_step", "max_replans"}),
        knob_validators={
            "max_iterations_per_step": _is_positive_int,
            "max_replans": _is_nonnegative_int,
        },
    ),
)
register_strategy(
    "rewoo",
    FlowStrategy(_build_rewoo, rewoo_initial_state, allowed_knobs=frozenset()),
)
register_strategy(
    "reflexion",
    FlowStrategy(
        _build_reflexion,
        reflexion_initial_state,
        allowed_knobs=frozenset({"max_retries", "threshold"}),
        knob_validators={"max_retries": _is_nonnegative_int, "threshold": _is_probability},
    ),
)
register_strategy(
    "generate_review",
    FlowStrategy(
        _build_generate_review,
        generate_review_initial_state,
        allowed_knobs=frozenset({"max_cycles", "max_review_iterations"}),
        knob_validators={
            "max_cycles": _is_positive_int,
            "max_review_iterations": _is_positive_int,
        },
    ),
)
register_strategy(
    "self_discovery",
    FlowStrategy(
        _build_self_discovery,
        self_discovery_initial_state,
        allowed_knobs=frozenset(),
    ),
)
register_strategy(
    "llm_compiler",
    FlowStrategy(
        _build_llm_compiler,
        llm_compiler_initial_state,
        allowed_knobs=frozenset({"max_replans"}),
        knob_validators={"max_replans": _is_nonnegative_int},
    ),
)
register_strategy(
    "tot",
    FlowStrategy(
        _build_tot,
        tot_initial_state,
        allowed_knobs=frozenset({"max_depth", "n_candidates", "search_strategy"}),
        knob_validators={
            "max_depth": _is_positive_int,
            "n_candidates": _is_positive_int,
            "search_strategy": _is_search_strategy,
        },
    ),
)
register_strategy(
    "lats",
    FlowStrategy(
        _build_lats,
        lats_initial_state,
        allowed_knobs=frozenset({"max_rollouts", "n_candidates"}),
        knob_validators={
            "max_rollouts": _is_positive_int,
            "n_candidates": _is_positive_int,
        },
    ),
)
