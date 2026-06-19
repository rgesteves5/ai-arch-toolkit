"""Canonical tool definition types.

Splits a tool into three concerns:

- ``ToolSchema``: the provider-facing contract (``name``/``description``/
  ``input_schema``). This — and only this — is sent to LLM APIs.
- ``ToolRuntimePolicy``: declarative governance metadata for a tool
  (capability, risk level, approval requirement). Never reaches a provider.
- ``ToolDefinition``: the runtime object binding a callable to its schema and
  policy. Produced by ``@tool`` and stored as ``fn.__tool_definition__``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

type RiskLevel = Literal["low", "medium", "high", "critical"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolSchema:
    """Provider-facing tool contract. The only part sent to LLM APIs."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)

    def to_provider_dict(self) -> dict[str, Any]:
        """Return the provider-safe dict (no governance metadata)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolRuntimePolicy:
    """Declarative governance metadata for a tool. Never sent to a provider."""

    capability: str | None = None
    risk_level: RiskLevel = "low"
    requires_approval: bool = False
    approval_reason: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolDefinition:
    """Canonical runtime object: callable + schema + runtime policy."""

    fn: Callable[..., Any]
    schema: ToolSchema
    policy: ToolRuntimePolicy = field(default_factory=ToolRuntimePolicy)
