"""Strategy builders — turn a ReasoningSpec into a runnable Flow.

A *strategy* is a named recipe for how an agent reasons. Each built-in strategy
adapts an existing Flow factory; consumers register their own with
``register_strategy``. A ``BuildContext`` separates serializable config
(``spec.knobs``) from runtime dependencies (``deps`` — an evaluator callable, a
second LLM, a memory store) that cannot live in a config file.

Multi-phase strategies accept canonical per-phase overrides through those two
buckets: runtime objects as deps (``planner_llm``, ``executor_tools``, …) and
prompts as knobs (``planner_system``, …), validated per strategy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
    serialized — an evaluator callable, a second LLM, a memory store. Built-in
    multi-phase strategies read canonical per-phase overrides from it
    (``planner_llm``, ``executor_tools``, …) and validate names and types.
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
    """Adapts a Flow factory plus an initial-state function into a builder.

    ``phases`` names the strategy's configurable phases, for introspection and
    static validation. ``allowed_deps``/``dep_validators`` mirror the knob
    machinery for runtime dependencies; ``allowed_deps=None`` (the default)
    skips dep validation, preserving behavior for user-registered strategies.
    """

    builder: Callable[[BuildContext], Flow]
    initializer: Callable[[Content], dict[str, Any]]
    supports_output_schema: bool = False
    allowed_knobs: frozenset[str] | None = None
    knob_validators: Mapping[str, Callable[[Any], bool]] = field(default_factory=dict)
    phases: frozenset[str] = frozenset()
    allowed_deps: frozenset[str] | None = None
    dep_validators: Mapping[str, Callable[[Any], bool]] = field(default_factory=dict)

    def build(self, ctx: BuildContext) -> Flow:
        self.validate_spec(ctx.spec)
        self._validate_deps(ctx)
        return self.builder(ctx)

    def validate_spec(self, spec: ReasoningSpec) -> None:
        """Validate a spec's knobs against this strategy without building it."""
        if self.allowed_knobs is not None:
            unknown = sorted(set(spec.knobs) - self.allowed_knobs)
            if unknown:
                raise ValueError(
                    f"strategy {spec.strategy!r} received unknown knobs: {', '.join(unknown)}"
                )
        for name, validator in self.knob_validators.items():
            if name in spec.knobs and not validator(spec.knobs[name]):
                raise ValueError(
                    f"strategy {spec.strategy!r} received invalid value "
                    f"for knob {name!r}: {spec.knobs[name]!r}"
                )

    def _validate_deps(self, ctx: BuildContext) -> None:
        if self.allowed_deps is not None:
            unknown = sorted(set(ctx.deps) - self.allowed_deps)
            if unknown:
                raise ValueError(
                    f"strategy {ctx.spec.strategy!r} received unknown deps: {', '.join(unknown)}"
                )
        for name, validator in self.dep_validators.items():
            if name in ctx.deps and not validator(ctx.deps[name]):
                raise ValueError(
                    f"strategy {ctx.spec.strategy!r} received invalid value for dep "
                    f"{name!r} of type {type(ctx.deps[name]).__name__}"
                )

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


def _is_llm(value: Any) -> bool:
    return isinstance(value, LLM)


def _is_tool_group(value: Any) -> bool:
    return isinstance(value, ToolGroup)


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _is_str_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return False
    return bool(value) and all(isinstance(item, str) and item for item in value)


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _knob_kwargs(spec: ReasoningSpec, mapping: Mapping[str, str]) -> dict[str, Any]:
    """Translate canonical phase knobs into factory kwargs, only when set."""
    return {target: spec.knobs[name] for name, target in mapping.items() if name in spec.knobs}


def _aliased_dep(ctx: BuildContext, canonical: str, legacy: str, default: Any) -> Any:
    """Read a dep by canonical name, accepting a documented legacy alias."""
    if canonical in ctx.deps and legacy in ctx.deps:
        raise ValueError(
            f"strategy {ctx.spec.strategy!r} received both {canonical!r} and its "
            f"legacy alias {legacy!r}; pass only one"
        )
    return ctx.deps.get(canonical, ctx.deps.get(legacy, default))


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
        llm_kwargs=dict(s.llm_kwargs) or None,
        planner_llm=ctx.deps.get("planner_llm"),
        exec_llm=ctx.deps.get("executor_llm"),
        exec_tools=ctx.deps.get("executor_tools"),
        solver_llm=ctx.deps.get("solver_llm"),
        **_knob_kwargs(s, {"planner_system": "planner_system", "solver_system": "solver_system"}),
    )


def _build_rewoo(ctx: BuildContext) -> Flow:
    s = ctx.spec
    return rewoo_flow(
        ctx.llm,
        ctx.tools,
        system=s.system,
        timeout=s.timeout,
        policy=s.policy,
        llm_kwargs=dict(s.llm_kwargs) or None,
        planner_llm=ctx.deps.get("planner_llm"),
        solver_llm=ctx.deps.get("solver_llm"),
        **_knob_kwargs(s, {"planner_system": "planner_system", "solver_system": "solver_system"}),
    )


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
        llm_kwargs=dict(s.llm_kwargs) or None,
        exec_llm=ctx.deps.get("executor_llm"),
        exec_tools=ctx.deps.get("executor_tools"),
        reflect_llm=ctx.deps.get("reflector_llm"),
        **_knob_kwargs(s, {"reflector_system": "reflect_system"}),
    )


def _build_generate_review(ctx: BuildContext) -> Flow:
    s = ctx.spec
    review_llm = _aliased_dep(ctx, "reviewer_llm", "review_llm", ctx.llm)
    review_tools = _aliased_dep(ctx, "reviewer_tools", "review_tools", ctx.tools)
    gen_kwargs = dict(s.llm_kwargs)
    if s.output_schema is not None:
        gen_kwargs.setdefault("output_schema", s.output_schema)
    # Global llm_kwargs apply to every phase; reviewer_kwargs wins per key.
    review_kwargs = {**dict(s.llm_kwargs), **dict(s.knobs.get("reviewer_kwargs") or {})}
    return generate_review_flow(
        gen_llm=ctx.deps.get("generator_llm", ctx.llm),
        review_llm=review_llm,
        gen_tools=ctx.deps.get("generator_tools", ctx.tools),
        review_tools=review_tools,
        gen_system=s.system,
        gen_kwargs=gen_kwargs or None,
        max_cycles=s.knobs.get("max_cycles", 3),
        max_gen_iterations=s.max_iterations,
        max_review_iterations=s.knobs.get("max_review_iterations", 5),
        timeout=s.timeout,
        policy=s.policy,
        review_kwargs=review_kwargs or None,
        **_knob_kwargs(s, {"reviewer_system": "review_system"}),
    )


def _build_self_discovery(ctx: BuildContext) -> Flow:
    s = ctx.spec
    kwargs = _knob_kwargs(
        s,
        {
            "select_system": "select_system",
            "adapt_system": "adapt_system",
            "plan_system": "plan_system",
            "solver_system": "solve_system",
        },
    )
    if "modules" in s.knobs:
        kwargs["modules"] = tuple(s.knobs["modules"])
    return self_discovery_flow(
        ctx.llm,
        ctx.tools,
        system=s.system,
        max_react_iterations=s.max_iterations,
        timeout=s.timeout,
        policy=s.policy,
        llm_kwargs=dict(s.llm_kwargs) or None,
        reasoning_llm=ctx.deps.get("reasoning_llm"),
        solver_llm=ctx.deps.get("solver_llm"),
        solver_tools=ctx.deps.get("solver_tools"),
        **kwargs,
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
        llm_kwargs=dict(s.llm_kwargs) or None,
        planner_llm=ctx.deps.get("planner_llm"),
        exec_llm=ctx.deps.get("executor_llm"),
        exec_tools=ctx.deps.get("executor_tools"),
        joiner_llm=ctx.deps.get("joiner_llm"),
        **_knob_kwargs(s, {"planner_system": "planner_system", "joiner_system": "joiner_system"}),
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
        llm_kwargs=dict(s.llm_kwargs) or None,
        gen_llm=ctx.deps.get("generator_llm"),
        eval_llm=ctx.deps.get("evaluator_llm"),
        solver_llm=ctx.deps.get("solver_llm"),
        **_knob_kwargs(s, {"evaluator_system": "evaluator_system"}),
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
        exploration_weight=s.knobs.get("exploration_weight", 1.41),
        max_react_iterations=s.max_iterations,
        evaluator_fn=evaluator_fn,
        timeout=s.timeout,
        policy=s.policy,
        llm_kwargs=dict(s.llm_kwargs) or None,
        rollout_llm=ctx.deps.get("rollout_llm"),
        rollout_tools=ctx.deps.get("rollout_tools"),
        eval_llm=ctx.deps.get("evaluator_llm"),
        solver_llm=ctx.deps.get("solver_llm"),
        reflector_llm=ctx.deps.get("reflector_llm"),
        **_knob_kwargs(
            s, {"evaluator_system": "evaluator_system", "reflector_system": "reflect_system"}
        ),
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
        allowed_deps=frozenset(),
    ),
)
register_strategy(
    "completion",
    FlowStrategy(
        _build_completion,
        _completion_initial_state,
        supports_output_schema=True,
        allowed_knobs=frozenset(),
        allowed_deps=frozenset(),
    ),
)
register_strategy(
    "plan_execute",
    FlowStrategy(
        _build_plan_execute,
        plan_execute_initial_state,
        allowed_knobs=frozenset(
            {"max_iterations_per_step", "max_replans", "planner_system", "solver_system"}
        ),
        knob_validators={
            "max_iterations_per_step": _is_positive_int,
            "max_replans": _is_nonnegative_int,
            "planner_system": _is_nonempty_str,
            "solver_system": _is_nonempty_str,
        },
        phases=frozenset({"planner", "executor", "solver"}),
        allowed_deps=frozenset({"planner_llm", "executor_llm", "executor_tools", "solver_llm"}),
        dep_validators={
            "planner_llm": _is_llm,
            "executor_llm": _is_llm,
            "executor_tools": _is_tool_group,
            "solver_llm": _is_llm,
        },
    ),
)
register_strategy(
    "rewoo",
    FlowStrategy(
        _build_rewoo,
        rewoo_initial_state,
        allowed_knobs=frozenset({"planner_system", "solver_system"}),
        knob_validators={
            "planner_system": _is_nonempty_str,
            "solver_system": _is_nonempty_str,
        },
        phases=frozenset({"planner", "solver"}),
        allowed_deps=frozenset({"planner_llm", "solver_llm"}),
        dep_validators={"planner_llm": _is_llm, "solver_llm": _is_llm},
    ),
)
register_strategy(
    "reflexion",
    FlowStrategy(
        _build_reflexion,
        reflexion_initial_state,
        allowed_knobs=frozenset({"max_retries", "reflector_system", "threshold"}),
        knob_validators={
            "max_retries": _is_nonnegative_int,
            "reflector_system": _is_nonempty_str,
            "threshold": _is_probability,
        },
        phases=frozenset({"executor", "reflector"}),
        allowed_deps=frozenset({"evaluator", "executor_llm", "executor_tools", "reflector_llm"}),
        dep_validators={
            "executor_llm": _is_llm,
            "executor_tools": _is_tool_group,
            "reflector_llm": _is_llm,
        },
    ),
)
register_strategy(
    "generate_review",
    FlowStrategy(
        _build_generate_review,
        generate_review_initial_state,
        supports_output_schema=True,
        allowed_knobs=frozenset(
            {"max_cycles", "max_review_iterations", "reviewer_kwargs", "reviewer_system"}
        ),
        knob_validators={
            "max_cycles": _is_positive_int,
            "max_review_iterations": _is_positive_int,
            "reviewer_kwargs": _is_mapping,
            "reviewer_system": _is_nonempty_str,
        },
        phases=frozenset({"generator", "reviewer"}),
        allowed_deps=frozenset(
            {
                "generator_llm",
                "generator_tools",
                "review_llm",
                "review_tools",
                "reviewer_llm",
                "reviewer_tools",
            }
        ),
        dep_validators={
            "generator_llm": _is_llm,
            "generator_tools": _is_tool_group,
            "review_llm": _is_llm,
            "review_tools": _is_tool_group,
            "reviewer_llm": _is_llm,
            "reviewer_tools": _is_tool_group,
        },
    ),
)
register_strategy(
    "self_discovery",
    FlowStrategy(
        _build_self_discovery,
        self_discovery_initial_state,
        allowed_knobs=frozenset(
            {"adapt_system", "modules", "plan_system", "select_system", "solver_system"}
        ),
        knob_validators={
            "adapt_system": _is_nonempty_str,
            "modules": _is_str_sequence,
            "plan_system": _is_nonempty_str,
            "select_system": _is_nonempty_str,
            "solver_system": _is_nonempty_str,
        },
        phases=frozenset({"reasoning", "solver"}),
        allowed_deps=frozenset({"reasoning_llm", "solver_llm", "solver_tools"}),
        dep_validators={
            "reasoning_llm": _is_llm,
            "solver_llm": _is_llm,
            "solver_tools": _is_tool_group,
        },
    ),
)
register_strategy(
    "llm_compiler",
    FlowStrategy(
        _build_llm_compiler,
        llm_compiler_initial_state,
        allowed_knobs=frozenset({"joiner_system", "max_replans", "planner_system"}),
        knob_validators={
            "joiner_system": _is_nonempty_str,
            "max_replans": _is_nonnegative_int,
            "planner_system": _is_nonempty_str,
        },
        phases=frozenset({"executor", "joiner", "planner"}),
        allowed_deps=frozenset({"executor_llm", "executor_tools", "joiner_llm", "planner_llm"}),
        dep_validators={
            "executor_llm": _is_llm,
            "executor_tools": _is_tool_group,
            "joiner_llm": _is_llm,
            "planner_llm": _is_llm,
        },
    ),
)
register_strategy(
    "tot",
    FlowStrategy(
        _build_tot,
        tot_initial_state,
        allowed_knobs=frozenset(
            {"evaluator_system", "max_depth", "n_candidates", "search_strategy"}
        ),
        knob_validators={
            "evaluator_system": _is_nonempty_str,
            "max_depth": _is_positive_int,
            "n_candidates": _is_positive_int,
            "search_strategy": _is_search_strategy,
        },
        phases=frozenset({"evaluator", "generator", "solver"}),
        allowed_deps=frozenset({"evaluator_llm", "generator_llm", "solver_llm"}),
        dep_validators={
            "evaluator_llm": _is_llm,
            "generator_llm": _is_llm,
            "solver_llm": _is_llm,
        },
    ),
)
register_strategy(
    "lats",
    FlowStrategy(
        _build_lats,
        lats_initial_state,
        allowed_knobs=frozenset(
            {
                "evaluator_system",
                "exploration_weight",
                "max_rollouts",
                "n_candidates",
                "reflector_system",
            }
        ),
        knob_validators={
            "evaluator_system": _is_nonempty_str,
            "exploration_weight": _is_positive_number,
            "max_rollouts": _is_positive_int,
            "n_candidates": _is_positive_int,
            "reflector_system": _is_nonempty_str,
        },
        phases=frozenset({"evaluator", "reflector", "rollout", "solver"}),
        allowed_deps=frozenset(
            {
                "evaluator_fn",
                "evaluator_llm",
                "reflector_llm",
                "rollout_llm",
                "rollout_tools",
                "solver_llm",
            }
        ),
        dep_validators={
            "evaluator_llm": _is_llm,
            "reflector_llm": _is_llm,
            "rollout_llm": _is_llm,
            "rollout_tools": _is_tool_group,
            "solver_llm": _is_llm,
        },
    ),
)
