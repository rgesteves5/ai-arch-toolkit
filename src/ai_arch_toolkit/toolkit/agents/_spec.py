"""ReasoningSpec — declarative description of how a single agent reasons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._response import OutputSchema

__all__ = ["ReasoningSpec"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ReasoningSpec:
    """Declarative description of how one agent reasons.

    Carries only the reasoning structure — the named ``strategy``, system
    prompt, and limits — not the model or tools, which are runtime objects
    supplied to ``build_flow``. ``knobs`` holds strategy-specific options.
    """

    strategy: str = "react"
    system: str = ""
    max_iterations: int = 10
    knobs: Mapping[str, Any] = field(default_factory=dict)
    policy: Policy | None = None
    timeout: float | None = None
    llm_kwargs: Mapping[str, Any] = field(default_factory=dict)
    output_schema: OutputSchema | type | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ReasoningSpec:
        """Build a spec from a plain mapping (e.g. parsed JSON/YAML/dict)."""
        policy = data.get("policy")
        return cls(
            strategy=str(data.get("strategy", "react")),
            system=str(data.get("system", "")),
            max_iterations=int(data.get("max_iterations", 10)),
            knobs=dict(data.get("knobs") or {}),
            policy=policy if isinstance(policy, Policy) else None,
            timeout=data.get("timeout"),
            llm_kwargs=dict(data.get("llm_kwargs") or {}),
            output_schema=_coerce_output_schema(data.get("output_schema")),
        )


def _coerce_output_schema(value: Any) -> OutputSchema | type | None:
    if value is None or isinstance(value, (OutputSchema, type)):
        return value
    if isinstance(value, Mapping):
        schema = value.get("schema")
        if not isinstance(schema, Mapping):
            return None
        return OutputSchema(
            name=str(value.get("name", "output")),
            schema=dict(schema),
            strict=bool(value.get("strict", True)),
        )
    return None
