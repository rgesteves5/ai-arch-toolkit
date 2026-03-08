"""Step — the unit of work, and Result — its output."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ai_arch_toolkit.core._response import Usage
from ai_arch_toolkit.core._state import StateSnapshot

if TYPE_CHECKING:
    from ai_arch_toolkit.core._policy import Policy

type StepFn = Callable[[StateSnapshot], Awaitable[Result]]


@dataclass(frozen=True, slots=True, kw_only=True)
class Result:
    """Output of a Step execution."""

    value: Any = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    confidence: float | None = None
    error: str | None = None
    duration: float = 0.0

    @property
    def is_ok(self) -> bool:
        return self.error is None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "artifacts": dict(self.artifacts),
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "cache_write_tokens": self.usage.cache_write_tokens,
                "cache_read_tokens": self.usage.cache_read_tokens,
            },
            "cost": self.cost,
            "confidence": self.confidence,
            "error": self.error,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Result:
        usage_data = data.get("usage", {})
        return cls(
            value=data.get("value"),
            artifacts=data.get("artifacts", {}),
            usage=Usage(**usage_data) if usage_data else Usage(),
            cost=data.get("cost", 0.0),
            confidence=data.get("confidence"),
            error=data.get("error"),
            duration=data.get("duration", 0.0),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Step:
    """A named unit of work with optional policy and scope."""

    name: str
    fn: StepFn
    policy: Policy | None = None
    scope: Any = None
    fallback: Step | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Step name must be non-empty")
