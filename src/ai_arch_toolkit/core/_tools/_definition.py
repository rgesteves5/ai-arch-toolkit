"""Canonical tool definition types.

Splits a tool into three concerns:

- ``ToolSchema``: the provider-facing contract (``name``/``description``/
  ``input_schema``). This — and only this — is sent to LLM APIs.
- ``ToolRuntimePolicy``: declarative governance metadata for a tool
  (capability, risk level, approval requirement, output and time bounds). Never reaches a
  provider.
- ``ToolDefinition``: the runtime object binding a callable to its schema and
  policy. Produced by ``@tool`` and stored as ``fn.__tool_definition__``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

type RiskLevel = Literal["low", "medium", "high", "critical"]

DEFAULT_MAX_OUTPUT_CHARS = 200_000
DEFAULT_TIMEOUT_S = 120.0


def check_bounds(max_output_chars: int | None, timeout_s: float | None) -> None:
    """Raise ``ValueError`` unless each bound is positive or ``None``."""
    for name, value in (("max_output_chars", max_output_chars), ("timeout_s", timeout_s)):
        if value is not None and value <= 0:
            msg = f"{name} must be positive or None, got {value!r}"
            raise ValueError(msg)


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
    """Declarative governance metadata for a tool. Never sent to a provider.

    Attributes:
        capability: What the tool reaches (``"network"``, ``"filesystem"``, ...), for gates and
            approval handlers.
        risk_level: How much harm a wrong call can do.
        requires_approval: Whether a human or handler must approve each call.
        approval_reason: Why approval is needed, shown to the approver.
        max_output_chars: Most characters of the result the model receives; the executor cuts the
            rest and marks the cut. ``None`` lets any size through.
        timeout_s: Seconds the executor waits for the tool before the call fails with
            ``timeout``. ``None`` waits as long as it takes. A synchronous tool cannot be killed:
            it runs in a daemon thread, which the executor stops waiting for.
    """

    capability: str | None = None
    risk_level: RiskLevel = "low"
    requires_approval: bool = False
    approval_reason: str = ""
    max_output_chars: int | None = DEFAULT_MAX_OUTPUT_CHARS
    timeout_s: float | None = DEFAULT_TIMEOUT_S

    def __post_init__(self) -> None:
        check_bounds(self.max_output_chars, self.timeout_s)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolDefinition:
    """Canonical runtime object: callable + schema + runtime policy."""

    fn: Callable[..., Any]
    schema: ToolSchema
    policy: ToolRuntimePolicy = field(default_factory=ToolRuntimePolicy)
