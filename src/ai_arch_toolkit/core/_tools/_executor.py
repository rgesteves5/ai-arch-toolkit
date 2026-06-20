"""Tool execution pipeline — one path, structured ``ToolResult`` everywhere."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from ai_arch_toolkit.core._redaction import Redactor
from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._approval import ApprovalHandler
from ai_arch_toolkit.core._tools._definition import ToolDefinition, ToolRuntimePolicy
from ai_arch_toolkit.core._tools._governance import (
    ApprovalGate,
    ExecutionContext,
    GateBlock,
    GateDryRun,
    RunState,
    ToolGate,
    default_redactor,
)
from ai_arch_toolkit.core._tools._result import ToolResult, _format_value
from ai_arch_toolkit.core._tools._schema import tool_schema

# --- Resolution ---------------------------------------------------------------


def _resolve_fn(tool_call: ToolCall, tools: list[Callable[..., Any]]) -> Callable[..., Any]:
    """Find the callable matching a tool call name.

    Matches by ``__tool_definition__.schema.name`` first, then falls back to
    ``__name__`` for plain (undecorated) callables.
    """
    for fn in tools:
        definition = getattr(fn, "__tool_definition__", None)
        if definition is not None and definition.schema.name == tool_call.name:
            return fn
    for fn in tools:
        if hasattr(fn, "__tool_definition__"):
            continue
        if getattr(fn, "__name__", None) == tool_call.name:
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
    """Convert an exception during tool execution to a structured, redacted result.

    Exception text is *redacted* (not hidden): the agent still sees useful
    messages like "backend down", but secret-shaped substrings are stripped.
    """
    message = redactor.redact_text(str(exc))
    if isinstance(exc, TypeError):
        return ToolResult.failure(
            "validation_error",
            f"Tool {tool_name!r} argument mismatch: {message}",
            details={"tool_name": tool_name},
        )
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


def _run_tool_sync(
    definition: ToolDefinition,
    tool_call: ToolCall,
    *,
    gates: Sequence[ToolGate],
    run_state: RunState,
    max_calls: int | None,
    redactor: Redactor,
) -> ToolResult:
    ctx = ExecutionContext(definition=definition, tool_call=tool_call)
    args = dict(tool_call.input)
    audit: dict[str, Any] = {}
    for gate in gates:
        result = gate.check_sync(ctx)
        if result is None:
            continue
        if isinstance(result, GateBlock):
            return _block_result(result, tool_call, audit, redactor)
        if isinstance(result, GateDryRun):
            return _dry_run_result(tool_call, {**audit, **result.audit}, redactor)
        args = result.args
        audit = {**audit, **result.audit}

    if max_calls is not None:
        if run_state.executed >= max_calls:
            return _max_calls_block(tool_call, max_calls, audit, redactor)
        run_state.executed += 1

    try:
        result_value = _coerce_result(definition.fn(**args))
    except Exception as exc:
        return _result_from_exception(tool_call.name, exc, redactor)
    return _with_audit(result_value, audit, redactor)


async def _arun_tool(
    definition: ToolDefinition,
    tool_call: ToolCall,
    *,
    gates: Sequence[ToolGate],
    run_state: RunState,
    max_calls: int | None,
    redactor: Redactor,
) -> ToolResult:
    ctx = ExecutionContext(definition=definition, tool_call=tool_call)
    args = dict(tool_call.input)
    audit: dict[str, Any] = {}
    for gate in gates:
        result = await gate.check(ctx)
        if result is None:
            continue
        if isinstance(result, GateBlock):
            return _block_result(result, tool_call, audit, redactor)
        if isinstance(result, GateDryRun):
            return _dry_run_result(tool_call, {**audit, **result.audit}, redactor)
        args = result.args
        audit = {**audit, **result.audit}

    if max_calls is not None:
        async with run_state.lock:
            if run_state.executed >= max_calls:
                return _max_calls_block(tool_call, max_calls, audit, redactor)
            run_state.executed += 1

    try:
        if inspect.iscoroutinefunction(definition.fn):
            result_value = _coerce_result(await definition.fn(**args))
        else:
            result_value = _coerce_result(await asyncio.to_thread(definition.fn, **args))
    except Exception as exc:
        return _result_from_exception(tool_call.name, exc, redactor)
    return _with_audit(result_value, audit, redactor)


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
