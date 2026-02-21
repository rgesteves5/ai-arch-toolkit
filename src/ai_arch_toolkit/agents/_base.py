"""Base agent types and abstract class."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from ai_arch_toolkit.llm._types import Response, ToolCall, ToolResult, Usage

StopReason = Literal[
    "completed",
    "timeout",
    "cancelled",
    "max_iterations",
    "budget_exhausted",
    "error",
]


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """An observable event fired during agent execution."""

    type: str  # "step_start", "step_end", "tool_call", "tool_result",
    # "error", "plan_created", "reflection"
    step_number: int = 0
    tool_name: str = ""
    tool_args: dict[str, object] = field(default_factory=dict)
    result: str = ""
    error: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Configuration for an agent run."""

    max_iterations: int = 10
    system: str = ""
    max_tokens: int | None = None
    planner_repair_retries: int = 1
    on_event: Callable[[AgentEvent], None] | None = None
    tool_choice: str | dict[str, object] | None = None
    parallel_tool_execution: bool = True
    timeout: float | None = None


@dataclass(frozen=True, slots=True)
class AgentStep:
    """A single step in an agent's execution."""

    step_number: int
    response: Response
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.usage.total_tokens == 0 and self.response.usage.total_tokens > 0:
            object.__setattr__(self, "usage", self.response.usage)


@dataclass(frozen=True, slots=True)
class BaseResult:
    """Base result shape returned by all agent runs."""

    answer: str
    steps: tuple[AgentStep, ...] = ()
    total_usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = "completed"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stop_reason == "completed":
            if self.answer == "[timeout exceeded]":
                object.__setattr__(self, "stop_reason", "timeout")
            elif self.answer == "[cancelled]":
                object.__setattr__(self, "stop_reason", "cancelled")
            elif self.answer == "[max iterations reached]":
                object.__setattr__(self, "stop_reason", "max_iterations")
            elif self.answer == "[token budget exceeded]":
                object.__setattr__(self, "stop_reason", "budget_exhausted")


@dataclass(frozen=True, slots=True)
class AgentResult(BaseResult):
    """Backward-compatible generic result."""


@dataclass(frozen=True, slots=True)
class ReActResult(AgentResult):
    """Result type for ReAct architecture."""


@dataclass(frozen=True, slots=True)
class PlanExecuteResult(AgentResult):
    """Result type for Plan-Execute architecture."""


@dataclass(frozen=True, slots=True)
class TreeOfThoughtsResult(AgentResult):
    """Result type for Tree-of-Thoughts architecture."""


@dataclass(frozen=True, slots=True)
class SelfDiscoveryResult(AgentResult):
    """Result type for Self-Discovery architecture."""


@dataclass(frozen=True, slots=True)
class LATSResult(AgentResult):
    """Result type for LATS architecture."""


@dataclass(frozen=True, slots=True)
class ReflexionResult(AgentResult):
    """Result type for Reflexion architecture."""


@dataclass(frozen=True, slots=True)
class ReWOOResult(AgentResult):
    """Result type for ReWOO architecture."""


@dataclass(frozen=True, slots=True)
class LLMCompilerResult(AgentResult):
    """Result type for LLMCompiler architecture."""


@dataclass(frozen=True, slots=True)
class CheckpointState:
    """Checkpoint state stub for future resume support."""

    agent_name: str
    task: str
    step_number: int
    stop_reason: str = ""
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class _BudgetTracker:
    """Simple per-run token budget tracker."""

    config: AgentConfig
    total_usage: Usage = field(default_factory=Usage)

    def observe_usage(self, usage: Usage) -> None:
        self.total_usage = _accumulate_usage(self.total_usage, usage)

    def exhausted_reason(self) -> StopReason | None:
        if (
            self.config.max_tokens is not None
            and self.total_usage.total_tokens >= self.config.max_tokens
        ):
            return "budget_exhausted"
        return None


def _accumulate_usage(total: Usage, delta: Usage) -> Usage:
    """Add two Usage objects together."""
    return Usage(
        input_tokens=total.input_tokens + delta.input_tokens,
        output_tokens=total.output_tokens + delta.output_tokens,
        total_tokens=total.total_tokens + delta.total_tokens,
        cache_creation_tokens=(total.cache_creation_tokens + delta.cache_creation_tokens),
        cache_read_tokens=total.cache_read_tokens + delta.cache_read_tokens,
    )


def _fire_event(config: AgentConfig, event_type: str, **kwargs: Any) -> None:
    """Fire an AgentEvent if an event handler is configured."""
    if config.on_event is not None:
        config.on_event(AgentEvent(type=event_type, **kwargs))


class BaseAgent(ABC):
    """Abstract base class for all agent architectures."""

    def __init__(
        self,
        client: Any,
        tools: Any,
        *,
        config: AgentConfig | None = None,
    ) -> None:
        self.client = client
        self.tools = tools
        self.config = config or AgentConfig()

    def _check_timeout(self, start: float) -> bool:
        """Return True if timeout has been exceeded."""
        return (
            self.config.timeout is not None and (time.monotonic() - start) >= self.config.timeout
        )

    def _resolve_cancellation_token(self, override: object | None) -> None:
        """Cancellation is currently disabled."""
        _ = override
        return None

    def _is_cancelled(self, token: object | None) -> bool:
        """Cancellation is currently disabled."""
        _ = token
        return False

    def _new_budget_manager(self) -> _BudgetTracker:
        return _BudgetTracker(config=self.config)

    def _observe_response(
        self,
        budget: _BudgetTracker,
        response: Response,
        *,
        step_number: int,
    ) -> float | None:
        _ = step_number
        budget.observe_usage(response.usage)
        return None

    def _finalize_result(
        self,
        result: BaseResult,
        *,
        result_type: type[BaseResult] = AgentResult,
    ) -> BaseResult:
        return result_type(
            answer=result.answer,
            steps=result.steps,
            total_usage=result.total_usage,
            stop_reason=result.stop_reason,
            metadata=result.metadata,
        )

    @abstractmethod
    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Execute the agent on a task and return the result."""
        ...

    async def async_run(self, task: str, **kwargs: Any) -> AgentResult:
        """Execute the agent asynchronously on a task and return the result."""
        raise NotImplementedError(f"{type(self).__name__} does not support async execution")

    def run_stream(self, task: str, **kwargs: Any) -> Iterator[AgentStep]:
        """Execute the agent, yielding each step as it completes."""
        result = self.run(task, stream=False, **kwargs)
        assert isinstance(result, BaseResult)
        yield from result.steps

    async def async_run_stream(self, task: str, **kwargs: Any) -> AsyncIterator[AgentStep]:
        """Execute the agent asynchronously, yielding each step."""
        result = await self.async_run(task, stream=False, **kwargs)
        assert isinstance(result, BaseResult)
        for step in result.steps:
            yield step

    def checkpoint(self, task: str) -> CheckpointState:
        """Checkpoint interface stub (persistence intentionally deferred)."""
        return CheckpointState(agent_name=type(self).__name__, task=task, step_number=0)

    def resume_from(self, checkpoint: CheckpointState, **kwargs: Any) -> BaseResult:
        """Resume interface stub (runtime resume intentionally deferred)."""
        msg = (
            "resume_from is not implemented yet. "
            "Checkpoint persistence/execution resume is deferred."
        )
        raise NotImplementedError(msg)
