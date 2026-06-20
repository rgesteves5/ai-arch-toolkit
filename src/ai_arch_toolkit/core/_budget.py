"""Cumulative runtime budget policy and tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._step import Result


@dataclass(frozen=True, slots=True, kw_only=True)
class BudgetPolicy:
    """Run-level cumulative budget limits."""

    max_wall_time: float | None = None
    max_llm_calls: int | None = None
    max_tool_calls: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_cost: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BudgetExceeded(Exception):
    """Raised when a cumulative budget limit is exceeded."""

    limit: str
    current: float
    maximum: float

    def __str__(self) -> str:
        return f"Budget exceeded for {self.limit}: {self.current} > {self.maximum}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "type": "budget_exceeded",
            "limit": self.limit,
            "current": self.current,
            "maximum": self.maximum,
            "message": str(self),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class BudgetState:
    """Cumulative budget usage for one flow run."""

    policy: BudgetPolicy
    started_at: float = field(default_factory=time.monotonic)
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    total_cost: float = 0.0
    exceeded: BudgetExceeded | None = None

    @classmethod
    def start(cls, policy: BudgetPolicy) -> BudgetState:
        """Create a new budget state."""
        return cls(policy=policy)

    @property
    def total_tokens(self) -> int:
        """Total billable-ish tokens tracked by the runtime."""
        return self.input_tokens + self.output_tokens

    @property
    def elapsed(self) -> float:
        """Elapsed wall time in seconds."""
        return time.monotonic() - self.started_at

    def check_wall_time(self) -> None:
        """Raise if the wall-time limit has been reached."""
        if self.policy.max_wall_time is not None and self.elapsed > self.policy.max_wall_time:
            raise BudgetExceeded(
                limit="wall_time",
                current=self.elapsed,
                maximum=self.policy.max_wall_time,
            )

    def check_llm_calls(self, count: int = 1) -> None:
        """Raise if adding LLM calls would exceed the budget."""
        self._check("llm_calls", self.llm_calls + count, self.policy.max_llm_calls)

    def check_tool_calls(self, count: int = 1) -> None:
        """Raise if adding tool calls would exceed the budget."""
        self._check("tool_calls", self.tool_calls + count, self.policy.max_tool_calls)

    def record_result(self, result: Result) -> BudgetState:
        """Record usage/cost/call counters reported by a step result."""
        return self.record(
            llm_calls=int(result.artifacts.get("budget_llm_calls", 0)),
            tool_calls=int(result.artifacts.get("budget_tool_calls", 0)),
            usage=result.usage,
            cost=result.cost,
        )

    def record_response(self, response: Response) -> BudgetState:
        """Record one LLM response."""
        return self.record(
            llm_calls=1,
            usage=response.usage,
            cost=response.cost or 0.0,
        )

    def record(
        self,
        *,
        llm_calls: int = 0,
        tool_calls: int = 0,
        usage: Usage | None = None,
        cost: float = 0.0,
    ) -> BudgetState:
        """Return a new state with added usage, raising if a limit is exceeded."""
        usage = usage or Usage()
        updated = BudgetState(
            policy=self.policy,
            started_at=self.started_at,
            llm_calls=self.llm_calls + llm_calls,
            tool_calls=self.tool_calls + tool_calls,
            input_tokens=self.input_tokens + usage.input_tokens,
            output_tokens=self.output_tokens + usage.output_tokens,
            cache_write_tokens=self.cache_write_tokens + usage.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens + usage.cache_read_tokens,
            total_cost=self.total_cost + cost,
            exceeded=self.exceeded,
        )
        updated.check_limits()
        return updated

    def with_exceeded(self, error: BudgetExceeded) -> BudgetState:
        """Return a copy annotated with the exceeded limit."""
        return BudgetState(
            policy=self.policy,
            started_at=self.started_at,
            llm_calls=self.llm_calls,
            tool_calls=self.tool_calls,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_write_tokens=self.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens,
            total_cost=self.total_cost,
            exceeded=error,
        )

    def check_limits(self) -> None:
        """Raise if any cumulative limit is exceeded."""
        self.check_wall_time()
        self._check("llm_calls", self.llm_calls, self.policy.max_llm_calls)
        self._check("tool_calls", self.tool_calls, self.policy.max_tool_calls)
        self._check("input_tokens", self.input_tokens, self.policy.max_input_tokens)
        self._check("output_tokens", self.output_tokens, self.policy.max_output_tokens)
        self._check("total_tokens", self.total_tokens, self.policy.max_total_tokens)
        self._check("total_cost", self.total_cost, self.policy.max_cost)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "elapsed": self.elapsed,
            "exceeded": self.exceeded.to_dict() if self.exceeded else None,
        }

    @staticmethod
    def _check(limit: str, current: float, maximum: float | None) -> None:
        if maximum is not None and current > maximum:
            raise BudgetExceeded(limit=limit, current=current, maximum=maximum)
