"""Human approval models for high-risk tool execution, and the preview of a call."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import inspect
import json
import logging
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._definition import RiskLevel, ToolDefinition, ToolPreview
from ai_arch_toolkit.core._tools._result import line_cut

logger = logging.getLogger(__name__)

# The most characters of a preview an approver or an audit receives: a hook's text, or the call's
# arguments as JSON (C07c, D64).
PREVIEW_MAX_CHARS = 16_000
_PREVIEW_CUT = "\n[preview cut at {kept} of {chars} characters]"
# The seconds a preview hook may take. Past them the preview is the call's arguments as JSON, and
# the hook, which cannot be stopped, is left to finish in its daemon thread (C07c, D64).
PREVIEW_TIMEOUT_S = 10.0

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


async def approval_request_for(tool_call: ToolCall, definition: ToolDefinition) -> ApprovalRequest:
    """The approval request for a call, from its tool's policy, with :func:`preview_for`'s
    preview: a tool's hook runs in a thread of its own, off the loop."""
    return _request(tool_call, definition, await preview_for(tool_call, definition))


def approval_request_for_sync(tool_call: ToolCall, definition: ToolDefinition) -> ApprovalRequest:
    """The approval request for a call, from its tool's policy, with :func:`preview_for`'s
    preview."""
    return _request(tool_call, definition, preview_for_sync(tool_call, definition))


def _request(tool_call: ToolCall, definition: ToolDefinition, preview: str) -> ApprovalRequest:
    policy = definition.policy
    return ApprovalRequest(
        tool_name=tool_call.name,
        arguments=dict(tool_call.input),
        capability=policy.capability,
        risk_level=policy.risk_level,
        preview=preview,
        reason=policy.approval_reason,
    )


async def preview_for(tool_call: ToolCall, definition: ToolDefinition) -> str:
    """:func:`preview_for_sync`, awaited: the tool's hook never holds the loop."""
    hook = definition.preview
    if hook is None:
        return _arguments_preview(tool_call)
    running = _started(hook, tool_call)
    waited = asyncio.wrap_future(running)
    try:
        await asyncio.wait((waited,), timeout=PREVIEW_TIMEOUT_S)
    finally:
        waited.cancel()  # once it is done, a no-op; before, the hook runs on, unheard
    return _answer(tool_call, running)


def preview_for_sync(tool_call: ToolCall, definition: ToolDefinition) -> str:
    """What the call will do, for a person: its tool's preview hook's text, or, for a tool
    without a hook, its name and arguments as JSON; either one cut at ``PREVIEW_MAX_CHARS`` with
    a note.

    The hook receives a copy of the call's arguments, as validated and changed by the gates
    before the one asking. It runs in a daemon thread of its own, in a copy of the caller's
    context, on this path as on the async one. A hook that raises, returns anything but text, or
    has not returned within ``PREVIEW_TIMEOUT_S`` seconds is logged, and the preview is the
    arguments as JSON, as for a tool without one; a hook that has not returned is left to finish
    on its own, and what it returns then is dropped.
    """
    hook = definition.preview
    if hook is None:
        return _arguments_preview(tool_call)
    running = _started(hook, tool_call)
    concurrent.futures.wait((running,), timeout=PREVIEW_TIMEOUT_S)
    return _answer(tool_call, running)


def _started(hook: ToolPreview, tool_call: ToolCall) -> concurrent.futures.Future[object]:
    """``hook`` called with a copy of the call's arguments, in a daemon thread of its own, inside
    a copy of the caller's context.

    A thread cannot be stopped, so a hook the caller stopped waiting for may still be running; a
    daemon thread never holds up ``asyncio.run`` or the end of the process (the executor runs a
    sync tool the same way, D30).
    """
    future: concurrent.futures.Future[object] = concurrent.futures.Future()
    context = contextvars.copy_context()
    arguments = dict(tool_call.input)

    def run() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(context.run(hook, arguments))
        except BaseException as exc:  # handed to whoever waits on the future
            future.set_exception(exc)

    threading.Thread(target=run, name="tool-preview", daemon=True).start()
    return future


def _answer(tool_call: ToolCall, running: concurrent.futures.Future[object]) -> str:
    """The preview from the hook's answer so far: its text cut to size, or the arguments
    preview (logged) when it has not returned, raised, or returned anything but text.

    Raises:
        BaseException: What the hook raised that is not an ``Exception`` (``KeyboardInterrupt``,
            ``SystemExit``), as if it had run in the caller's thread.
    """
    if not running.done():
        logger.warning(
            "preview of tool %r did not return within %gs; showing its arguments",
            tool_call.name,
            PREVIEW_TIMEOUT_S,
        )
        return _arguments_preview(tool_call)
    error = running.exception()
    if error is not None and not isinstance(error, Exception):
        raise error
    if error is not None:
        logger.warning(
            "preview of tool %r raised; showing its arguments", tool_call.name, exc_info=error
        )
        return _arguments_preview(tool_call)
    text = running.result()
    if not isinstance(text, str):
        if inspect.iscoroutine(text):
            text.close()
        logger.warning(
            "preview of tool %r returned a %s, not text; showing its arguments",
            tool_call.name,
            type(text).__name__,
        )
        return _arguments_preview(tool_call)
    return _bounded(text)


def _bounded(text: str) -> str:
    """``text``, or its first ``PREVIEW_MAX_CHARS`` characters (ending on a line where one lies
    in their second half) and a note that says so."""
    if len(text) <= PREVIEW_MAX_CHARS:
        return text
    kept = line_cut(text, 0, PREVIEW_MAX_CHARS)
    return text[:kept] + _PREVIEW_CUT.format(kept=kept, chars=len(text))


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


def _arguments_preview(tool_call: ToolCall) -> str:
    """The call as its tool's name and its arguments as JSON, cut at ``PREVIEW_MAX_CHARS``: the
    arguments themselves stay whole in the request."""
    try:
        args = json.dumps(tool_call.input, sort_keys=True)
    except TypeError:
        args = repr(tool_call.input)
    return _bounded(f"{tool_call.name}({args})")
