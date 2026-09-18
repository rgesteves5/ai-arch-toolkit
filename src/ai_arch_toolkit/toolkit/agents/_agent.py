"""Agent — a configured reasoning unit you can run, stream, or compose."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ai_arch_toolkit.core._content import Content, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._metering._scope import RunConfig
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._step import Step
from ai_arch_toolkit.core._sync import _run_sync
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents._compile import build_flow, initial_state, read_answer
from ai_arch_toolkit.toolkit.agents._spec import ReasoningSpec
from ai_arch_toolkit.toolkit.agents.flows._keys import MESSAGES
from ai_arch_toolkit.toolkit.budget import BudgetPolicy, BudgetReport
from ai_arch_toolkit.toolkit.flow._executor import FlowExecution
from ai_arch_toolkit.toolkit.flow._flow import Flow, FlowEvent, FlowResult

__all__ = ["Agent", "AgentExecution", "AgentResult"]


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
        self._flow = build_flow(spec, llm, tools if tools is not None else ToolGroup(), deps=deps)
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
            agent._make_state = lambda task: {MESSAGES: [user(task)]}
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
        self,
        task: Content,
        *,
        budget_policy: BudgetPolicy | None = None,
        config: RunConfig | None = None,
    ) -> AgentResult:
        """Run the agent on one task and return a structured result.

        A per-run ``budget_policy`` caps this run (overriding any budget baked into
        the backing flow). A per-run ``config`` fully specifies the run's meter — sinks,
        redactor, pricer, retained events, controller — and takes precedence over
        ``budget_policy``; put a ``BudgetController`` in it to keep a budget. Both are
        ignored when the agent runs nested under an enclosing metered scope, which
        shares one cumulative budget.
        """
        state = State(operational=self._make_state(task))
        flow_result = await self._flow.run(state, budget_policy=budget_policy, config=config)
        return _agent_result(flow_result)

    def run_sync(
        self,
        task: Content,
        *,
        budget_policy: BudgetPolicy | None = None,
        config: RunConfig | None = None,
    ) -> AgentResult:
        """Synchronous wrapper for ``run``."""
        return _run_sync(self.run(task, budget_policy=budget_policy, config=config))

    def iter(
        self,
        task: Content,
        *,
        budget_policy: BudgetPolicy | None = None,
        config: RunConfig | None = None,
    ) -> AgentExecution:
        """Iterate one task: flow events as they happen, then ``.result`` (an ``AgentResult``).

        ``budget_policy`` and ``config`` work as in ``run``.
        """
        state = State(operational=self._make_state(task))
        return AgentExecution(self._flow.iter(state, budget_policy=budget_policy, config=config))

    def as_step(self) -> Step:
        """Wrap this agent's Flow as a Step for composition into a larger Flow."""
        return self._flow.as_step()


class AgentExecution:
    """An agent run being iterated: flow events as they happen, then its :class:`AgentResult`.

    Behaves like :class:`~ai_arch_toolkit.toolkit.flow.FlowExecution`: nothing runs until the first
    event is requested, ``result`` is ``None`` until the run has finished, and a ``break`` does not
    stop a run that is still referenced — :meth:`aclose` or ``async with`` does.
    """

    __slots__ = ("_execution", "_result")

    def __init__(self, execution: FlowExecution) -> None:
        self._execution = execution
        self._result: AgentResult | None = None

    def __aiter__(self) -> AgentExecution:
        return self

    async def __anext__(self) -> FlowEvent:
        return await self._execution.__anext__()

    async def aclose(self) -> None:
        """Stop the run: cancel the steps still running and close its meter."""
        await self._execution.aclose()

    async def __aenter__(self) -> AgentExecution:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    @property
    def result(self) -> AgentResult | None:
        """The finished run's result, or ``None`` while it is running or if it was abandoned."""
        if self._result is None and self._execution.result is not None:
            self._result = _agent_result(self._execution.result)
        return self._result


def _agent_result(flow_result: FlowResult) -> AgentResult:
    text, response = read_answer(flow_result.state, flow_result)
    # A cyclic flow can execute the same named step more than once. ``results`` keeps only the
    # latest result per name, so derive errors from the trace or an earlier failed turn would
    # disappear after a later success.
    errors = tuple(record.error for record in flow_result.trace.steps if record.error)
    report = flow_result.meter  # meter-derived (single source of truth); snapshot once, reuse
    return AgentResult(
        text=text,
        response=response,
        flow_result=flow_result,
        usage=flow_result.usage,
        cost=report.cost if report is not None else 0.0,
        report=report,
        errors=errors,
    )
