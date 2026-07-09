"""Agent — a configured reasoning unit you can run, stream, or compose."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ai_arch_toolkit.core._content import Content, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._step import Step
from ai_arch_toolkit.core._sync import _run_sync
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents._compile import build_flow, extract_text, initial_state
from ai_arch_toolkit.toolkit.agents._spec import ReasoningSpec
from ai_arch_toolkit.toolkit.budget import BudgetPolicy, BudgetReport
from ai_arch_toolkit.toolkit.flow._flow import Flow, FlowEvent, FlowResult

__all__ = ["Agent", "AgentResult"]


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentResult:
    """Outcome of one Agent run. ``usage``/``cost``/``report`` are meter-derived (the truth)."""

    text: str
    response: Response | None
    flow_result: FlowResult
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    report: BudgetReport | None = None
    errors: tuple[str, ...] = ()


class Agent:
    """A configured reasoning unit: a ReasoningSpec bound to an LLM and tools.

    The compiled Flow is built once and reused across ``run`` calls. For a Flow
    you built yourself (any composition of Steps), use ``Agent.from_flow``.
    """

    __slots__ = ("_flow", "_make_state")

    def __init__(
        self,
        spec: ReasoningSpec,
        llm: LLM,
        tools: ToolGroup | None = None,
        *,
        deps: Mapping[str, Any] | None = None,
    ) -> None:
        self._flow = build_flow(spec, llm, tools or ToolGroup(), deps=deps)
        self._make_state: Callable[[Content], dict[str, Any]] = lambda task: initial_state(
            spec, task
        )

    @classmethod
    def from_flow(
        cls,
        flow: Flow,
        *,
        init_state: dict[str, Any] | Callable[[Content], dict[str, Any]] | None = None,
    ) -> Agent:
        """Wrap an arbitrary Flow as an Agent (the escape hatch for full freedom).

        ``init_state`` builds the per-task operational state: a callable mapping
        the task to a dict, a fixed dict (task ignored), or ``None`` for the
        default ``{"messages": [user(task)]}``.
        """
        agent = cls.__new__(cls)
        agent._flow = flow
        if init_state is None:
            agent._make_state = lambda task: {"messages": [user(task)]}
        elif callable(init_state):
            agent._make_state = init_state
        else:
            fixed = dict(init_state)
            agent._make_state = lambda _task: dict(fixed)
        return agent

    @property
    def flow(self) -> Flow:
        """The compiled Flow backing this agent."""
        return self._flow

    async def run(
        self, task: Content, *, budget_policy: BudgetPolicy | None = None
    ) -> AgentResult:
        """Run the agent on one task and return a structured result.

        A per-run ``budget_policy`` caps this run (overriding any budget baked into
        the backing flow); it is ignored when the agent runs nested under an
        enclosing metered scope, which shares one cumulative budget.
        """
        state = State(operational=self._make_state(task))
        flow_result = await self._flow.run(state, budget_policy=budget_policy)
        response = state.get("response") or state.get("last_response")
        if not isinstance(response, Response):
            response = None
        errors = tuple(r.error for r in flow_result.results.values() if r.error)
        report = flow_result.meter  # meter-derived (single source of truth); snapshot once, reuse
        return AgentResult(
            text=extract_text(state, flow_result),
            response=response,
            flow_result=flow_result,
            usage=flow_result.usage,
            cost=report.cost if report is not None else 0.0,
            report=report,
            errors=errors,
        )

    def run_sync(self, task: Content, *, budget_policy: BudgetPolicy | None = None) -> AgentResult:
        """Synchronous wrapper for ``run``."""
        return _run_sync(self.run(task, budget_policy=budget_policy))

    async def iter(
        self, task: Content, *, budget_policy: BudgetPolicy | None = None
    ) -> AsyncIterator[FlowEvent]:
        """Stream flow events while running on one task (``budget_policy`` as in ``run``)."""
        state = State(operational=self._make_state(task))
        async for event in self._flow.iter(state, budget_policy=budget_policy):
            yield event

    def as_step(self) -> Step:
        """Wrap this agent's Flow as a Step for composition into a larger Flow."""
        return self._flow.as_step()
