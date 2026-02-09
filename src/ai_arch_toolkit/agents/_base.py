"""Base agent types and abstract class."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ai_arch_toolkit.llm._types import Response, ToolCall, ToolResult, Usage


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


@dataclass(frozen=True, slots=True)
class AgentResult:
    """The final result of an agent run."""

    answer: str
    steps: tuple[AgentStep, ...] = ()
    total_usage: Usage = field(default_factory=Usage)


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

    @abstractmethod
    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Execute the agent on a task and return the result."""
        ...

    async def async_run(self, task: str, **kwargs: Any) -> AgentResult:
        """Execute the agent asynchronously on a task and return the result."""
        raise NotImplementedError(f"{type(self).__name__} does not support async execution")

    def run_stream(self, task: str, **kwargs: Any) -> Iterator[AgentStep]:
        """Execute the agent, yielding each step as it completes."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming execution")
        yield  # pragma: no cover — unreachable, satisfies generator typing

    async def async_run_stream(self, task: str, **kwargs: Any) -> AsyncIterator[AgentStep]:
        """Execute the agent asynchronously, yielding each step."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support async streaming execution"
        )
        yield  # pragma: no cover — unreachable, satisfies generator typing
