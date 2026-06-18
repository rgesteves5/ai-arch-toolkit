"""Trace — execution history for debugging and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ai_arch_toolkit.core._redaction import RedactionPolicy, Redactor, TraceMode
from ai_arch_toolkit.core._response import Usage

type PolicyDecision = Literal[
    "retry",
    "fallback",
    "timeout",
    "budget_exceeded",
    "cost_exceeded",
    "low_confidence",
    "escalate",
    "halt",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class StepTrace:
    """Execution record for a single Step."""

    name: str
    input_state: dict[str, Any] = field(default_factory=dict)
    output_result: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    cost: float = 0.0
    confidence: float | None = None
    usage: Usage = field(default_factory=Usage)
    attempts: int = 1
    policy_decisions: tuple[PolicyDecision, ...] = ()
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    children: tuple[StepTrace, ...] = ()
    started_at: float = 0.0

    def to_dict(
        self,
        *,
        trace_mode: TraceMode = "redacted",
        redactor: Redactor | None = None,
    ) -> dict[str, Any]:
        active_redactor = redactor or Redactor(RedactionPolicy(trace_mode=trace_mode))
        if trace_mode == "metadata_only":
            input_state: dict[str, Any] = {}
            output_result: dict[str, Any] = {}
            error = active_redactor.redact_text(self.error) if self.error else None
        else:
            input_state = active_redactor.redact(self.input_state)
            output_result = active_redactor.redact(self.output_result)
            error = active_redactor.redact_text(self.error) if self.error else None
        return {
            "name": self.name,
            "input_state": input_state,
            "output_result": output_result,
            "duration": self.duration,
            "cost": self.cost,
            "confidence": self.confidence,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "cache_write_tokens": self.usage.cache_write_tokens,
                "cache_read_tokens": self.usage.cache_read_tokens,
            },
            "attempts": self.attempts,
            "policy_decisions": list(self.policy_decisions),
            "error": error,
            "skipped": self.skipped,
            "skip_reason": active_redactor.redact_text(self.skip_reason)
            if self.skip_reason
            else None,
            "children": [
                c.to_dict(trace_mode=trace_mode, redactor=active_redactor) for c in self.children
            ],
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepTrace:
        usage_data = data.get("usage", {})
        return cls(
            name=data["name"],
            input_state=data.get("input_state", {}),
            output_result=data.get("output_result", {}),
            duration=data.get("duration", 0.0),
            cost=data.get("cost", 0.0),
            confidence=data.get("confidence"),
            usage=Usage(**usage_data) if usage_data else Usage(),
            attempts=data.get("attempts", 1),
            policy_decisions=tuple(data.get("policy_decisions", ())),
            error=data.get("error"),
            skipped=data.get("skipped", False),
            skip_reason=data.get("skip_reason"),
            children=tuple(StepTrace.from_dict(c) for c in data.get("children", ())),
            started_at=data.get("started_at", 0.0),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Trace:
    """Execution record for a complete Flow."""

    flow_name: str
    steps: tuple[StepTrace, ...] = ()
    initial_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- Navigation ---

    def step(self, name: str) -> StepTrace | None:
        """Find a StepTrace by name, searching recursively."""
        return self._find(name, self.steps)

    def flow(self, name: str) -> StepTrace | None:
        """Find a nested flow trace by name (steps with children)."""
        for st in self._iter_all(self.steps):
            if st.name == name and st.children:
                return st
        return None

    # --- Aggregates ---

    @property
    def total_cost(self) -> float:
        return sum(st.cost for st in self._iter_all(self.steps))

    @property
    def total_duration(self) -> float:
        return self.duration

    @property
    def confidence(self) -> float | None:
        """Minimum confidence across all non-skipped steps."""
        values = [
            st.confidence
            for st in self._iter_all(self.steps)
            if st.confidence is not None and not st.skipped
        ]
        return min(values) if values else None

    @property
    def total_usage(self) -> Usage:
        all_steps = list(self._iter_all(self.steps))
        return Usage(
            input_tokens=sum(st.usage.input_tokens for st in all_steps),
            output_tokens=sum(st.usage.output_tokens for st in all_steps),
            cache_write_tokens=sum(st.usage.cache_write_tokens for st in all_steps),
            cache_read_tokens=sum(st.usage.cache_read_tokens for st in all_steps),
        )

    # --- Serialization ---

    def to_dict(
        self,
        *,
        trace_mode: TraceMode = "redacted",
        redactor: Redactor | None = None,
    ) -> dict[str, Any]:
        active_redactor = redactor or Redactor(RedactionPolicy(trace_mode=trace_mode))
        initial_state = (
            {} if trace_mode == "metadata_only" else active_redactor.redact(self.initial_state)
        )
        return {
            "flow_name": self.flow_name,
            "trace_mode": trace_mode,
            "steps": [
                s.to_dict(trace_mode=trace_mode, redactor=active_redactor) for s in self.steps
            ],
            "initial_state": initial_state,
            "duration": self.duration,
            "metadata": active_redactor.redact(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        return cls(
            flow_name=data["flow_name"],
            steps=tuple(StepTrace.from_dict(s) for s in data.get("steps", ())),
            initial_state=data.get("initial_state", {}),
            duration=data.get("duration", 0.0),
            metadata=data.get("metadata", {}),
        )

    # --- Internal ---

    @staticmethod
    def _find(name: str, steps: tuple[StepTrace, ...]) -> StepTrace | None:
        for st in steps:
            if st.name == name:
                return st
            found = Trace._find(name, st.children)
            if found is not None:
                return found
        return None

    @staticmethod
    def _iter_all(steps: tuple[StepTrace, ...]) -> list[StepTrace]:
        """Flatten all step traces including children."""
        result: list[StepTrace] = []
        for st in steps:
            result.append(st)
            result.extend(Trace._iter_all(st.children))
        return result
