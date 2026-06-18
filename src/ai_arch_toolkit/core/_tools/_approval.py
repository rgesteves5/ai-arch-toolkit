"""Human approval models for high-risk tool execution."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ai_arch_toolkit.core._response import ToolCall

type RiskLevel = Literal["low", "medium", "high", "critical"]
type ApprovalHandler = Callable[
    ["ApprovalRequest"], "ApprovalDecision | Awaitable[ApprovalDecision]"
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ApprovalRequest:
    """Request emitted before executing a tool that requires approval."""

    tool_name: str
    arguments: dict[str, Any]
    capability: str | None = None
    risk_level: RiskLevel = "low"
    preview: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "capability": self.capability,
            "risk_level": self.risk_level,
            "preview": self.preview,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ApprovalDecision:
    """Decision returned by a human or external approval handler."""

    approved: bool = False
    denied: bool = False
    modified_args: dict[str, Any] | None = None
    reviewer: str | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.approved and self.denied:
            raise ValueError("ApprovalDecision cannot be both approved and denied")

    @classmethod
    def approve(
        cls,
        *,
        modified_args: dict[str, Any] | None = None,
        reviewer: str | None = None,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalDecision:
        """Create an approval decision."""
        return cls(
            approved=True,
            modified_args=modified_args,
            reviewer=reviewer,
            reason=reason,
            metadata=metadata or {},
        )

    @classmethod
    def deny(
        cls,
        *,
        reviewer: str | None = None,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalDecision:
        """Create a denial decision."""
        return cls(
            denied=True,
            reviewer=reviewer,
            reason=reason,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "approved": self.approved,
            "denied": self.denied,
            "modified_args": self.modified_args,
            "reviewer": self.reviewer,
            "reason": self.reason,
            "metadata": self.metadata,
        }


def approval_metadata(
    *,
    capability: str | None = None,
    risk_level: RiskLevel = "low",
    requires_approval: bool = False,
    approval_reason: str = "",
) -> dict[str, Any]:
    """Build the runtime approval metadata stored on a tool definition."""
    return {
        "capability": capability,
        "risk_level": risk_level,
        "requires_approval": requires_approval,
        "approval_reason": approval_reason,
    }


def approval_request_for(tool_call: ToolCall, tool_def: dict[str, Any]) -> ApprovalRequest:
    """Create an approval request for a tool call."""
    return ApprovalRequest(
        tool_name=tool_call.name,
        arguments=dict(tool_call.input),
        capability=tool_def.get("capability"),
        risk_level=tool_def.get("risk_level", "low"),
        preview=_preview(tool_call),
        reason=tool_def.get("approval_reason", ""),
    )


async def resolve_approval(
    request: ApprovalRequest,
    handler: ApprovalHandler | None,
) -> ApprovalDecision:
    """Resolve an approval request asynchronously, denying by default."""
    if handler is None:
        return ApprovalDecision.deny(reason="No approval handler configured")
    decision = handler(request)
    if inspect.isawaitable(decision):
        return await decision
    return decision


def resolve_approval_sync(
    request: ApprovalRequest,
    handler: ApprovalHandler | None,
) -> ApprovalDecision:
    """Resolve an approval request synchronously, denying by default."""
    if handler is None:
        return ApprovalDecision.deny(reason="No approval handler configured")
    decision = handler(request)
    if inspect.isawaitable(decision):
        if inspect.iscoroutine(decision):
            decision.close()
        return ApprovalDecision.deny(reason="Synchronous execution cannot await approval handler")
    return decision


def _preview(tool_call: ToolCall) -> str:
    try:
        args = json.dumps(tool_call.input, sort_keys=True)
    except TypeError:
        args = repr(tool_call.input)
    return f"{tool_call.name}({args})"
