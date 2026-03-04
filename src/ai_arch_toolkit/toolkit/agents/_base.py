"""Agent types and BaseAgent ABC — async-first with sync wrappers."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, overload

from ai_arch_toolkit.core._content import Content, tool_result
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import Attempt, OutputSchema, Response, ToolCall, Usage
from ai_arch_toolkit.core._sync import _run_sync, _stream_sync
from ai_arch_toolkit.core._tools._group import ToolGroup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

type StopReason = Literal["completed", "max_iterations", "timeout", "budget_exhausted", "error"]

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PhaseConfig:
    """Optional per-phase overrides for LLM and tools."""

    llm: LLM | None = None
    tools: ToolGroup | None = None


def _resolve_llm(phase: PhaseConfig | None, default: LLM) -> LLM:
    """Return the phase-specific LLM or the default."""
    return phase.llm if phase is not None and phase.llm is not None else default


def _resolve_tools(phase: PhaseConfig | None, default: ToolGroup) -> ToolGroup:
    """Return the phase-specific ToolGroup or the default."""
    return phase.tools if phase is not None and phase.tools is not None else default


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentConfig:
    """Configuration shared by all agent architectures."""

    max_iterations: int = 10
    system: str = ""
    max_tokens: int | None = None
    timeout: float | None = None
    tool_choice: str | None = None
    parallel_tool_calls: bool = True
    on_event: Callable[[AgentEvent], None] | None = None
    llm_kwargs: dict[str, Any] = field(default_factory=dict)
    output_schema: OutputSchema | type | None = None

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {self.max_iterations}")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentEvent:
    """A single observable event emitted during an agent run."""

    type: Literal["step_start", "step_end", "tool_call", "tool_result", "error"]
    step: int = 0
    tool_name: str = ""
    tool_call_id: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    error: str = ""
    response: Response | None = None
    stop_reason: StopReason | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentStep:
    """One iteration of an agent's reasoning loop."""

    step: int
    response: Response
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[dict[str, Any], ...] = ()
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentResult:
    """Final output of an agent run."""

    answer: str
    parsed: Any = None
    steps: tuple[AgentStep, ...] = ()
    total_usage: Usage = field(default_factory=Usage)
    total_cost: float = 0.0
    stop_reason: StopReason = "completed"
    all_attempts: tuple[Attempt, ...] = ()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_usage(a: Usage, b: Usage) -> Usage:
    """Sum two frozen Usage objects."""
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cache_write_tokens=a.cache_write_tokens + b.cache_write_tokens,
        cache_read_tokens=a.cache_read_tokens + b.cache_read_tokens,
    )


# ---------------------------------------------------------------------------
# BaseAgent ABC
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Abstract base for all agent architectures.

    Subclasses implement ``_run_loop()`` as an async generator yielding
    ``AgentEvent`` objects.  ``run()`` / ``run_sync()`` handle consumption,
    result collection, and sync bridging.

    ``_run_loop()`` is a pure generator — it must NOT call ``_fire()``.
    Callback dispatch happens in ``_consume()`` (for ``stream=False``) or
    is the caller's responsibility (for ``stream=True``).
    """

    __slots__ = ("config", "llm", "tools")

    def __init__(
        self,
        llm: LLM,
        tools: ToolGroup,
        *,
        config: AgentConfig | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.config = config or AgentConfig()

    @abstractmethod
    async def _run_loop(self, task: Content, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        """Core loop — yield AgentEvent objects.

        The final ``step_end`` event MUST have ``stop_reason`` set.
        Must not call ``_fire()`` — the consumer handles callbacks.
        """
        yield  # pragma: no cover — make this a valid async generator
        ...  # pragma: no cover

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    @overload
    async def run(
        self, task: Content, *, stream: Literal[False] = ..., **kwargs: Any
    ) -> AgentResult: ...

    @overload
    async def run(
        self, task: Content, *, stream: Literal[True], **kwargs: Any
    ) -> AsyncIterator[AgentEvent]: ...

    async def run(
        self, task: Content, *, stream: bool = False, **kwargs: Any
    ) -> AgentResult | AsyncIterator[AgentEvent]:
        if stream:
            return self._run_loop(task, **kwargs)
        return await self._consume(self._run_loop(task, **kwargs))

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    @overload
    def run_sync(
        self, task: Content, *, stream: Literal[False] = ..., **kwargs: Any
    ) -> AgentResult: ...

    @overload
    def run_sync(
        self, task: Content, *, stream: Literal[True], **kwargs: Any
    ) -> Iterator[AgentEvent]: ...

    def run_sync(
        self, task: Content, *, stream: bool = False, **kwargs: Any
    ) -> AgentResult | Iterator[AgentEvent]:
        if stream:
            return _stream_sync(lambda: self._run_loop(task, **kwargs))
        return _run_sync(self.run(task, **kwargs))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fire(self, event: AgentEvent) -> None:
        """Fire on_event callback if configured."""
        cb = self.config.on_event
        if cb is not None:
            cb(event)

    def _check_timeout(self, start: float) -> bool:
        """Return True if timeout exceeded."""
        if self.config.timeout is None:
            return False
        return (time.monotonic() - start) > self.config.timeout

    def _check_budget(self, total_usage: Usage) -> bool:
        """Return True if token budget exceeded."""
        if self.config.max_tokens is None:
            return False
        total = total_usage.input_tokens + total_usage.output_tokens
        return total >= self.config.max_tokens

    async def _consume(self, aiter: AsyncIterator[AgentEvent]) -> AgentResult:
        """Drain event stream → AgentResult, firing callbacks along the way."""
        steps: list[AgentStep] = []
        total_usage = Usage()
        total_cost = 0.0
        all_attempts: list[Attempt] = []
        stop_reason: StopReason = "completed"
        answer = ""

        # Accumulate per-step state
        current_tcs: list[ToolCall] = []
        current_tr: list[dict[str, Any]] = []

        async for event in aiter:
            self._fire(event)

            if event.type == "step_start":
                logger.info("Agent step %d started", event.step)
            elif event.type == "step_end":
                logger.info("Agent step %d ended reason=%s", event.step, event.stop_reason)
                if event.stop_reason is not None:
                    stop_reason = event.stop_reason
                if event.response is not None:
                    resp = event.response
                    total_usage = _add_usage(total_usage, resp.usage)
                    total_cost += resp.cost or 0.0
                    all_attempts.extend(resp.attempts)
                    steps.append(
                        AgentStep(
                            step=event.step,
                            response=resp,
                            tool_calls=tuple(current_tcs),
                            tool_results=tuple(current_tr),
                            usage=resp.usage,
                        )
                    )
                    answer = resp.text
                    current_tcs = []
                    current_tr = []
            elif event.type == "tool_call":
                current_tcs.append(
                    ToolCall(id=event.tool_call_id, name=event.tool_name, input=event.tool_args)
                )
            elif event.type == "tool_result":
                current_tr.append(
                    tool_result(
                        event.result,
                        tool_use_id=event.tool_call_id,
                        name=event.tool_name,
                    )
                )

        parsed = None
        if steps and steps[-1].response:
            parsed = steps[-1].response.parsed

        return AgentResult(
            answer=answer,
            parsed=parsed,
            steps=tuple(steps),
            total_usage=total_usage,
            total_cost=total_cost,
            stop_reason=stop_reason,
            all_attempts=tuple(all_attempts),
        )
