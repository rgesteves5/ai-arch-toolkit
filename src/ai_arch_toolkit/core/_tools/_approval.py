"""Human approval models for high-risk tool execution."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._definition import RiskLevel, ToolRuntimePolicy

type ApprovalStatus = Literal["approved", "denied"]
type ApprovalHandler = Callable[
    ["ApprovalRequest"], "ApprovalDecision | Awaitable[ApprovalDecision]"
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ApprovalRequest:
    """Request emitted before executing a tool that requires approval.

    The handler receives the *unredacted* arguments — it needs the real values
    to make a decision. Redaction is applied only to what gets stored in audit
    metadata, never to this request.
    """

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
    """Decision returned by a human or external approval handler.

    The decision is a single ``status`` — ``"approved"`` or ``"denied"``. There
    is no ambiguous "neither" state to misinterpret; construct via
    :meth:`approve` / :meth:`deny`.
    """

    status: ApprovalStatus
    modified_args: dict[str, Any] | None = None
    reviewer: str | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        """True when the decision approves execution."""
        return self.status == "approved"

    @property
    def denied(self) -> bool:
        """True when the decision denies execution."""
        return self.status == "denied"

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
            status="approved",
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
            status="denied",
            reviewer=reviewer,
            reason=reason,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "status": self.status,
            "approved": self.approved,
            "denied": self.denied,
            "modified_args": self.modified_args,
            "reviewer": self.reviewer,
            "reason": self.reason,
            "metadata": self.metadata,
        }


def approval_request_for(tool_call: ToolCall, policy: ToolRuntimePolicy) -> ApprovalRequest:
    """Create an approval request for a tool call from its runtime policy."""
    return ApprovalRequest(
        tool_name=tool_call.name,
        arguments=dict(tool_call.input),
        capability=policy.capability,
        risk_level=policy.risk_level,
        preview=_preview(tool_call),
        reason=policy.approval_reason,
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
