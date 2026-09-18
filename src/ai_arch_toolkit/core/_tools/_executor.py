"""Tool execution pipeline — one path, structured ``ToolResult`` everywhere."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import functools
import inspect
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ai_arch_toolkit.core._metering._admission import AdmissionDenied
from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._operation import MeterOperation, OperationRequest
from ai_arch_toolkit.core._metering._scope import current_meter, current_span_id
from ai_arch_toolkit.core._redaction import Redactor
from ai_arch_toolkit.core._response import ToolCall, Usage
from ai_arch_toolkit.core._sync import _run_sync
from ai_arch_toolkit.core._tools._approval import ApprovalHandler
from ai_arch_toolkit.core._tools._definition import ToolDefinition, ToolRuntimePolicy
from ai_arch_toolkit.core._tools._governance import (
    ApprovalGate,
    ExecutionContext,
    GateBlock,
    GateDryRun,
    GateResult,
    RunState,
    ToolGate,
    default_redactor,
)
from ai_arch_toolkit.core._tools._result import ToolResult, _format_value
from ai_arch_toolkit.core._tools._schema import callable_name, tool_schema
from ai_arch_toolkit.core._tools._validation import (
    ArgumentError,
    bind_arguments,
    validate_arguments,
)

logger = logging.getLogger(__name__)

# --- Resolution ---------------------------------------------------------------


def _resolve_fn(tool_call: ToolCall, tools: list[Callable[..., Any]]) -> Callable[..., Any]:
    """Find the callable matching a tool call name.

    Matches by ``__tool_definition__.schema.name`` first, then falls back to the
    function name for plain (undecorated) callables, a ``functools.partial``'s included.
    """
    for fn in tools:
        definition = getattr(fn, "__tool_definition__", None)
        if definition is not None and definition.schema.name == tool_call.name:
            return fn
    for fn in tools:
        if hasattr(fn, "__tool_definition__"):
            continue
        if callable_name(fn) == tool_call.name:
            return fn
    msg = f"Unknown tool: {tool_call.name!r}"
    raise KeyError(msg)


def _definition_for(fn: Callable[..., Any]) -> ToolDefinition:
    """Return the canonical ``ToolDefinition`` for a callable.

    Decorated functions carry one; plain callables get a synthesized definition
    with an inferred schema and a default (low-risk, no-approval) policy.
    """
    definition = getattr(fn, "__tool_definition__", None)
    if definition is not None:
        return definition
    return ToolDefinition(fn=fn, schema=tool_schema(fn), policy=ToolRuntimePolicy())


def _resolve_definition(tool_call: ToolCall, tools: list[Callable[..., Any]]) -> ToolDefinition:
    return _definition_for(_resolve_fn(tool_call, tools))


# --- Invocation ---------------------------------------------------------------


class _TimedOut(Exception):
    """The executor stopped waiting for a tool."""


def _in_thread(call: Callable[[], Any]) -> concurrent.futures.Future[Any]:
    """Run ``call`` in a daemon thread of its own, inside a copy of the caller's context.

    A thread cannot be killed, so a tool the executor stopped waiting for may still be running.
    Unlike the default executor's threads, a daemon thread never holds up ``asyncio.run`` or the
    end of the process.
    """
    future: concurrent.futures.Future[Any] = concurrent.futures.Future()
    context = contextvars.copy_context()

    def run() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(context.run(call))
        except BaseException as exc:  # handed to whoever waits on the future
            future.set_exception(exc)

    threading.Thread(target=run, name="tool", daemon=True).start()
    return future


async def _call(fn: Callable[..., Any], positional: list[Any], keywords: dict[str, Any]) -> Any:
    """Call a tool from async code: coroutine functions on the loop, the rest in a thread.

    A sync function that returns an awaitable has it awaited here.
    """
    if inspect.iscoroutinefunction(fn):
        return await fn(*positional, **keywords)
    thread = _in_thread(functools.partial(fn, *positional, **keywords))
    value = await asyncio.wrap_future(thread)
    if inspect.isawaitable(value):
        return await value
    return value


async def _invoke(
    fn: Callable[..., Any],
    positional: list[Any],
    keywords: dict[str, Any],
    timeout_s: float | None,
) -> Any:
    """Await the tool for at most ``timeout_s``; a coroutine tool is cancelled at the deadline.

    Raises:
        _TimedOut: The deadline passed. A ``TimeoutError`` the tool raised itself propagates.
    """
    deadline = asyncio.timeout(timeout_s)
    try:
        async with deadline:
            return await _call(fn, positional, keywords)
    except TimeoutError:
        if deadline.expired():
            raise _TimedOut from None
        raise


# --- Limits -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Limits:
    """Bounds on one call. As a group's ceiling, each bound only ever tightens the tool's."""

    max_output_chars: int | None = None
    timeout_s: float | None = None

    def over(self, policy: ToolRuntimePolicy) -> _Limits:
        """The stricter of this ceiling and the tool's policy, bound by bound."""
        return _Limits(
            _stricter(self.max_output_chars, policy.max_output_chars),
            _stricter(self.timeout_s, policy.timeout_s),
        )


_NO_CEILING = _Limits()


def _stricter[N: (int, float)](ceiling: N | None, own: N | None) -> N | None:
    if ceiling is None:
        return own
    if own is None:
        return ceiling
    return min(ceiling, own)


_TRUNCATION_NOTE = "\n\n[Output truncated: kept {kept} of {chars} characters.]"


def _bounded(result: ToolResult, max_output_chars: int | None) -> ToolResult:
    """``result`` with the text the model reads cut to ``max_output_chars``, and the cut noted.

    A value (structured or not) becomes its cut model text; an error keeps its type and has its
    message cut.
    """
    text = result.to_model_text()
    if max_output_chars is None or len(text) <= max_output_chars:
        return result
    note = _TRUNCATION_NOTE.format(kept=max_output_chars, chars=len(text))
    metadata = {**result.metadata, "truncated": {"chars": len(text), "kept": max_output_chars}}
    if result.error is None:
        return replace(result, value=text[:max_output_chars] + note, metadata=metadata)
    message = result.error.message[:max_output_chars] + note
    return replace(result, error=replace(result.error, message=message), metadata=metadata)


def _timed_out(tool_call: ToolCall, timeout_s: float | None) -> ToolResult:
    return ToolResult.failure(
        "timeout",
        f"Tool {tool_call.name!r} did not finish within {timeout_s:g}s",
        retryable=True,
        details={"tool_name": tool_call.name, "timeout_s": timeout_s},
    )


def _validated(definition: ToolDefinition, tool_call: ToolCall, arguments: Any) -> ToolCall:
    """The call with its arguments validated and coerced against the tool's schema.

    Raises:
        ArgumentError: The arguments don't fit the schema.
    """
    coerced = validate_arguments(definition.fn, definition.schema.input_schema, arguments)
    return replace(tool_call, input=coerced)


def _validation_failure(
    tool_call: ToolCall, error: ArgumentError, audit: dict[str, Any], redactor: Redactor
) -> ToolResult:
    details: dict[str, Any] = {"tool_name": tool_call.name}
    if error.argument is not None:
        details["argument"] = error.argument
    result = ToolResult.failure(
        "validation_error",
        f"Tool {tool_call.name!r} {redactor.redact_text(str(error))}",
        details=details,
    )
    return _with_audit(result, audit, redactor)


# --- Result helpers -----------------------------------------------------------


def _format_result(result: Any) -> str:
    """Convert a tool result to string for LLM consumption."""
    if isinstance(result, ToolResult):
        return result.to_model_text()
    return _format_value(result)


def _coerce_result(value: Any) -> ToolResult:
    """Normalize a tool return value into a ToolResult."""
    if isinstance(value, ToolResult):
        return value
    return ToolResult.success(value)


def _result_from_exception(tool_name: str, exc: Exception, redactor: Redactor) -> ToolResult:
    """Convert an exception raised by a tool to a structured, redacted result.

    Arguments were validated and bound before the call, so any exception here — ``TypeError``
    included — comes from the tool itself. Exception text is *redacted* (not hidden): the agent
    still sees useful messages like "backend down", but secret-shaped substrings are stripped.
    """
    message = redactor.redact_text(str(exc))
    return ToolResult.failure(
        "runtime_error",
        message,
        retryable=True,
        details={"tool_name": tool_name, "exception_type": type(exc).__name__},
    )


def _with_audit(result: ToolResult, audit: dict[str, Any], redactor: Redactor) -> ToolResult:
    """Attach redaction-safe audit metadata under ``metadata['audit']``."""
    if not audit:
        return result
    redacted = redactor.redact(audit)
    existing = result.metadata.get("audit", {})
    return replace(result, metadata={**result.metadata, "audit": {**existing, **redacted}})


def _block_result(
    block: GateBlock, tool_call: ToolCall, audit: dict[str, Any], redactor: Redactor
) -> ToolResult:
    result = ToolResult.failure(
        block.error_type,
        block.message,
        retryable=block.retryable,
        safe_to_show=block.safe_to_show,
        details={"tool_name": tool_call.name},
    )
    return _with_audit(result, {**audit, **block.audit}, redactor)


def _dry_run_result(tool_call: ToolCall, audit: dict[str, Any], redactor: Redactor) -> ToolResult:
    result = ToolResult.success(
        f"[dry-run] would call {tool_call.name}",
        metadata={"governance": {"outcome": "dry_run", "executed": False}},
    )
    return _with_audit(result, audit, redactor)


def _max_calls_block(
    tool_call: ToolCall, limit: int, audit: dict[str, Any], redactor: Redactor
) -> ToolResult:
    return _block_result(
        GateBlock(
            error_type="max_calls_exceeded",
            message=f"Tool blocked by governance: max tool calls exceeded ({limit}).",
        ),
        tool_call,
        audit,
        redactor,
    )


# --- Pipeline -----------------------------------------------------------------

_NO_USAGE = Usage()  # tools consume no tokens
_ZERO_COST = Cost.known(Money.zero())


def _meter_tool_open(tool_call: ToolCall) -> tuple[MeterOperation | None, Cost]:
    """Open + start a metered op for a tool that is about to run, or ``(None, …)`` if unmetered.

    Called only after gates + max-calls pass, so a blocked/dry-run tool is never metered.
    ``AdmissionDenied`` from ``open`` propagates (terminal — a tool executor never converts it
    to a ``ToolResult``; only the flow executor does).
    """
    scope = current_meter()
    if scope is None:
        return None, _ZERO_COST
    request = OperationRequest(
        kind="tool",
        parent_span_id=current_span_id() or scope.run_span_id,
        metadata={"tool": tool_call.name},
    )
    op = scope.open(request)
    op.mark_started()
    # A tool has no token cost, so it's free unless a custom pricer says otherwise. A pricer that
    # RAISES or returns an estimate (settle rejects estimates) must not flip a successful tool into
    # an error OR leak this started op — fall back to free and keep going.
    cost = _ZERO_COST
    if scope.pricer is not None:
        try:
            priced = scope.pricer.price(request, _NO_USAGE)
        except Exception:
            logger.exception("pricer %r raised pricing a tool; recording it free", scope.pricer)
            priced = _ZERO_COST
        if priced.kind != "estimated":
            cost = priced
    return op, cost


@dataclass(frozen=True, slots=True)
class _Admitted:
    """A call that passed validation and every gate, with its arguments bound."""

    tool_call: ToolCall
    audit: dict[str, Any]
    positional: list[Any]
    keywords: dict[str, Any]


def _apply(
    decision: GateResult | None,
    definition: ToolDefinition,
    current: ToolCall,
    audit: dict[str, Any],
    redactor: Redactor,
) -> tuple[ToolCall, dict[str, Any]] | ToolResult:
    """One gate's decision: the call as the gate left it, or the result that stops it."""
    if decision is None:
        return current, audit
    if isinstance(decision, GateBlock):
        return _block_result(decision, current, audit, redactor)
    if isinstance(decision, GateDryRun):
        return _dry_run_result(current, {**audit, **decision.audit}, redactor)
    audit = {**audit, **decision.audit}
    try:
        return _validated(definition, current, decision.args), audit
    except ArgumentError as error:
        return _validation_failure(current, error, audit, redactor)


def _bind(
    definition: ToolDefinition, current: ToolCall, audit: dict[str, Any], redactor: Redactor
) -> _Admitted | ToolResult:
    try:
        positional, keywords = bind_arguments(definition.fn, current.input)
    except ArgumentError as error:
        return _validation_failure(current, error, audit, redactor)
    return _Admitted(current, audit, positional, keywords)


def _admit_sync(
    definition: ToolDefinition,
    tool_call: ToolCall,
    gates: Sequence[ToolGate],
    redactor: Redactor,
) -> _Admitted | ToolResult:
    """Validate, then pass every gate (each sees the arguments as the previous one left them)."""
    try:
        current = _validated(definition, tool_call, tool_call.input)
    except ArgumentError as error:
        return _validation_failure(tool_call, error, {}, redactor)
    audit: dict[str, Any] = {}
    for gate in gates:
        decision = gate.check_sync(ExecutionContext(definition=definition, tool_call=current))
        step = _apply(decision, definition, current, audit, redactor)
        if isinstance(step, ToolResult):
            return step
        current, audit = step
    return _bind(definition, current, audit, redactor)


async def _admit(
    definition: ToolDefinition,
    tool_call: ToolCall,
    gates: Sequence[ToolGate],
    redactor: Redactor,
) -> _Admitted | ToolResult:
    """Validate, then pass every gate (each sees the arguments as the previous one left them)."""
    try:
        current = _validated(definition, tool_call, tool_call.input)
    except ArgumentError as error:
        return _validation_failure(tool_call, error, {}, redactor)
    audit: dict[str, Any] = {}
    for gate in gates:
        decision = await gate.check(ExecutionContext(definition=definition, tool_call=current))
        step = _apply(decision, definition, current, audit, redactor)
        if isinstance(step, ToolResult):
            return step
        current, audit = step
    return _bind(definition, current, audit, redactor)


def _spent(run_state: RunState, max_calls: int | None) -> int | None:
    """Count one call against ``max_calls``; the limit itself when it was already reached."""
    if max_calls is None:
        return None
    if run_state.executed >= max_calls:
        return max_calls
    run_state.executed += 1
    return None


def _finished(
    op: MeterOperation | None, cost: Cost, admitted: _Admitted, value: Any, redactor: Redactor
) -> ToolResult:
    result = _with_audit(_coerce_result(value), admitted.audit, redactor)
    if op is not None:
        op.settle(usage=_NO_USAGE, cost=cost)
    return result


def _failed(
    op: MeterOperation | None,
    exc: BaseException,
    admitted: _Admitted,
    limits: _Limits,
    redactor: Redactor,
) -> ToolResult:
    """The result of a tool that raised or ran out of time; the op keeps its count, costs nothing.

    A budget denial is terminal and a cancellation is not the tool's: both propagate.
    """
    if op is not None:
        op.fail("unbilled")
    if isinstance(exc, _TimedOut):
        return _timed_out(admitted.tool_call, limits.timeout_s)
    if isinstance(exc, AdmissionDenied) or not isinstance(exc, Exception):
        raise exc
    return _result_from_exception(admitted.tool_call.name, exc, redactor)


def _run_tool_sync(
    definition: ToolDefinition,
    tool_call: ToolCall,
    *,
    gates: Sequence[ToolGate],
    run_state: RunState,
    max_calls: int | None,
    redactor: Redactor,
    ceiling: _Limits = _NO_CEILING,
) -> ToolResult:
    limits = ceiling.over(definition.policy)
    admitted = _admit_sync(definition, tool_call, gates, redactor)
    if isinstance(admitted, ToolResult):
        return _bounded(admitted, limits.max_output_chars)
    if (limit := _spent(run_state, max_calls)) is not None:
        return _max_calls_block(admitted.tool_call, limit, admitted.audit, redactor)
    op, cost = _meter_tool_open(admitted.tool_call)  # AdmissionDenied here is terminal
    fn, args = definition.fn, (admitted.positional, admitted.keywords)
    try:
        # A loop of the sync path's own: the daemon thread of a timed-out tool does not hold it.
        value = _run_sync(_invoke(fn, *args, limits.timeout_s))
        result = _finished(op, cost, admitted, value, redactor)
    except BaseException as exc:
        result = _failed(op, exc, admitted, limits, redactor)
    return _bounded(result, limits.max_output_chars)


async def _arun_tool(
    definition: ToolDefinition,
    tool_call: ToolCall,
    *,
    gates: Sequence[ToolGate],
    run_state: RunState,
    max_calls: int | None,
    redactor: Redactor,
    ceiling: _Limits = _NO_CEILING,
) -> ToolResult:
    limits = ceiling.over(definition.policy)
    admitted = await _admit(definition, tool_call, gates, redactor)
    if isinstance(admitted, ToolResult):
        return _bounded(admitted, limits.max_output_chars)
    limit = None
    if max_calls is not None:
        async with run_state.lock:
            limit = _spent(run_state, max_calls)
    if limit is not None:
        return _max_calls_block(admitted.tool_call, limit, admitted.audit, redactor)
    op, cost = _meter_tool_open(admitted.tool_call)  # AdmissionDenied here is terminal
    fn, args = definition.fn, (admitted.positional, admitted.keywords)
    try:
        value = await _invoke(fn, *args, limits.timeout_s)
        result = _finished(op, cost, admitted, value, redactor)
    except BaseException as exc:
        result = _failed(op, exc, admitted, limits, redactor)
    return _bounded(result, limits.max_output_chars)


# --- Public free functions ----------------------------------------------------


def execute_tool(
    tool_call: ToolCall,
    tools: list[Callable[..., Any]],
    *,
    approval_handler: ApprovalHandler | None = None,
) -> ToolResult:
    """Execute a tool call synchronously, returning a structured ``ToolResult``."""
    try:
        definition = _resolve_definition(tool_call, tools)
    except KeyError:
        return ToolResult.failure(
            "unknown_tool",
            f"Unknown tool: {tool_call.name!r}",
            details={"tool_name": tool_call.name},
        )
    return _run_tool_sync(
        definition,
        tool_call,
        gates=(ApprovalGate(approval_handler),),
        run_state=RunState(),
        max_calls=None,
        redactor=default_redactor(),
    )


async def async_execute_tool(
    tool_call: ToolCall,
    tools: list[Callable[..., Any]],
    *,
    approval_handler: ApprovalHandler | None = None,
) -> ToolResult:
    """Execute a tool call asynchronously, returning a structured ``ToolResult``."""
    try:
        definition = _resolve_definition(tool_call, tools)
    except KeyError:
        return ToolResult.failure(
            "unknown_tool",
            f"Unknown tool: {tool_call.name!r}",
            details={"tool_name": tool_call.name},
        )
    return await _arun_tool(
        definition,
        tool_call,
        gates=(ApprovalGate(approval_handler),),
        run_state=RunState(),
        max_calls=None,
        redactor=default_redactor(),
    )
