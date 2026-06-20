"""Runtime governance for tool execution.

A small set of stateless *gates* run before a tool executes; each may block,
modify arguments, or request a dry run. The call-count budget is intentionally
*not* a gate — it is a stateful, concurrency-sensitive counter enforced
atomically at the commit step of the pipeline (see ``_executor``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from ai_arch_toolkit.core._redaction import RedactionPolicy, Redactor
from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._approval import (
    ApprovalHandler,
    approval_request_for,
    resolve_approval,
    resolve_approval_sync,
)
from ai_arch_toolkit.core._tools._definition import ToolDefinition

type GovernanceOutcome = Literal[
    "executed",
    "approval_denied",
    "dangerous_tool_blocked",
    "max_calls_exceeded",
    "dry_run",
    "budget_exceeded",
]


def default_redactor() -> Redactor:
    """Return the default redactor used for audit metadata."""
    return Redactor(RedactionPolicy(trace_mode="redacted"))


@dataclass(slots=True)
class RunState:
    """Per-run execution state.

    Owns the call-count budget counter and the lock guarding it. A ``ToolGroup``
    creates one ``RunState`` per instance; reuse across runs requires
    :meth:`reset`. The lock is created lazily so a ``RunState`` can be built
    outside a running event loop.
    """

    executed: int = 0
    _lock: asyncio.Lock | None = None

    def reset(self) -> None:
        """Reset the call-count budget for a fresh run."""
        self.executed = 0

    @property
    def lock(self) -> asyncio.Lock:
        """Return the budget lock, creating it on first use."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionContext:
    """Inputs available to a gate when deciding on a tool call."""

    definition: ToolDefinition
    tool_call: ToolCall


@dataclass(frozen=True, slots=True, kw_only=True)
class GateBlock:
    """A gate result that blocks execution with a structured failure."""

    error_type: GovernanceOutcome
    message: str
    safe_to_show: bool = True
    retryable: bool = False
    audit: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class GateModify:
    """A gate result that allows execution, possibly with modified arguments."""

    args: dict[str, Any]
    audit: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class GateDryRun:
    """A gate result that reports what would run without executing."""

    audit: dict[str, Any] = field(default_factory=dict)


type GateResult = GateBlock | GateModify | GateDryRun


@runtime_checkable
class ToolGate(Protocol):
    """A pre-execution governance gate.

    ``check`` / ``check_sync`` return ``None`` to pass through, or a
    :data:`GateResult` to short-circuit. Gates must be stateless so they are
    safe under concurrent execution.
    """

    def check_sync(self, ctx: ExecutionContext) -> GateResult | None: ...

    async def check(self, ctx: ExecutionContext) -> GateResult | None: ...


class DangerousToolGate:
    """Block tools whose name is marked dangerous unless explicitly allowed."""

    __slots__ = ("_allow", "_blocked")

    def __init__(self, *, blocked: Iterable[str], allow: bool = False) -> None:
        self._blocked = frozenset(blocked)
        self._allow = allow

    def check_sync(self, ctx: ExecutionContext) -> GateResult | None:
        if not self._allow and ctx.tool_call.name in self._blocked:
            return GateBlock(
                error_type="dangerous_tool_blocked",
                message=(
                    f"Tool blocked by governance: {ctx.tool_call.name!r} "
                    "requires --allow-dangerous-tools."
                ),
            )
        return None

    async def check(self, ctx: ExecutionContext) -> GateResult | None:
        return self.check_sync(ctx)


class DryRunGate:
    """Short-circuit every call as a dry run, recording the arguments in audit."""

    __slots__ = ("_dry_run",)

    def __init__(self, *, dry_run: bool = True) -> None:
        self._dry_run = dry_run

    def check_sync(self, ctx: ExecutionContext) -> GateResult | None:
        if self._dry_run:
            return GateDryRun(audit={"arguments": dict(ctx.tool_call.input)})
        return None

    async def check(self, ctx: ExecutionContext) -> GateResult | None:
        return self.check_sync(ctx)


class ApprovalGate:
    """Require human/external approval for tools whose policy demands it.

    No-ops for tools that do not require approval. Denies by default when no
    handler is configured.
    """

    __slots__ = ("_handler",)

    def __init__(self, handler: ApprovalHandler | None = None) -> None:
        self._handler = handler

    def check_sync(self, ctx: ExecutionContext) -> GateResult | None:
        if not ctx.definition.policy.requires_approval:
            return None
        request = approval_request_for(ctx.tool_call, ctx.definition.policy)
        decision = resolve_approval_sync(request, self._handler)
        return self._outcome(ctx, request, decision)

    async def check(self, ctx: ExecutionContext) -> GateResult | None:
        if not ctx.definition.policy.requires_approval:
            return None
        request = approval_request_for(ctx.tool_call, ctx.definition.policy)
        decision = await resolve_approval(request, self._handler)
        return self._outcome(ctx, request, decision)

    @staticmethod
    def _outcome(ctx: ExecutionContext, request: Any, decision: Any) -> GateResult:
        audit = {"approval": {"request": request.to_dict(), "decision": decision.to_dict()}}
        if decision.denied:
            return GateBlock(
                error_type="approval_denied",
                message=f"Tool {ctx.tool_call.name!r} requires approval and was denied",
                audit=audit,
            )
        return GateModify(
            args=decision.modified_args or dict(ctx.tool_call.input),
            audit=audit,
        )
