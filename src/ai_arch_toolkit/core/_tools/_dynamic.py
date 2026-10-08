"""Tools built at run time from a complete JSON Schema, for definitions that arrive as data (D62).

An MCP server, an OpenAPI document or the app itself describes a tool by name, description and
JSON Schema, with no Python signature behind it. :func:`tool_from_schema` turns that into a tool
like any ``@tool`` function: the same callable carrying ``__tool_definition__``, which a
``ToolGroup``, ``prepare_tools``, ``execute_tool`` and ``run_tools`` take as they are, so the
call goes through the one governed executor (validation, gates, approval, ``max_calls``,
metering) and reaches each provider through its adapter.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterator, Mapping

from ai_arch_toolkit.core._tools._definition import (
    ToolDefinition,
    ToolPreview,
    ToolRuntimePolicy,
    ToolSchema,
)
from ai_arch_toolkit.core._tools._schema import _SCHEMA_DEPTH_LIMIT, _inline_local_refs

# Receives a call's arguments in one dict; may be ``async``.
type ToolHandler = Callable[[dict[str, object]], object]


def tool_from_schema(
    handler: ToolHandler,
    /,
    *,
    name: str,
    description: str = "",
    input_schema: Mapping[str, object],
    policy: ToolRuntimePolicy | None = None,
    preview: ToolPreview | None = None,
) -> Callable[..., object]:
    """Build a tool from a complete JSON Schema and a handler that takes the arguments as a dict.

    The tool is a callable that takes the arguments by keyword and calls
    ``handler(dict(arguments))``, so a key need not be a Python name (``"x-api-version"``,
    ``"from"``). Before the handler runs, the executor checks the arguments against the schema
    and coerces them as it does for ``@tool`` (see the ``Argument validation`` docs): numeric and
    boolean strings at the root, ``anyOf``/``oneOf`` and ``type`` lists, ``additionalProperties:
    false`` at the root; nested values pass as they came, for the handler (or the server behind
    it) to check. An ``async def`` handler runs on the caller's loop, a synchronous one in a
    thread. It returns the tool's value, or a :class:`ToolResult`, which passes intact; one that
    cannot answer raises :class:`ToolFailure` with its type, and any other exception becomes a
    ``runtime_error`` with its message redacted.

    The tool keeps a copy of ``input_schema`` and sends it as it came (``title``, ``$schema``,
    ``x-*`` keys included), with its local ``#/$defs/...`` references inlined; a recursive one
    stays a reference, with its ``$defs`` table.

    Args:
        handler: Receives each call's arguments in one dict.
        name: The name the model calls the tool by.
        description: What the tool does, for the model.
        input_schema: A JSON Schema whose root is ``{"type": "object", ...}``.
        policy: Governance metadata (capability, risk, approval, output and time bounds); the
            default is a low-risk tool that needs no approval.
        preview: Writes what a call will do, from its arguments, for the approver and a dry run
            (see :class:`ToolDefinition`); ``None`` shows the arguments as JSON.

    Raises:
        TypeError: ``preview`` is not a plain (not ``async``) function.
        ValueError: ``name`` is not portable (1 to 64 letters, digits, ``_`` or ``-``, starting
            with a letter or ``_``); ``input_schema`` is not JSON (``NaN``, a set), its root is
            not an object, it refers outside itself (``https://...``, another file) or to a
            definition that is not an object schema (``true``, a list), it nests deeper than 100
            levels, or inlining its references would make it larger than about 100,000
            characters or deeper than 100 levels (each reference followed counts as one). No
            other exception escapes for a schema that is JSON.
    """
    schema = ToolSchema(
        name=name, description=description, input_schema=_standalone(name, input_schema)
    )
    fn = _calling(handler)
    fn.__name__ = fn.__qualname__ = name
    fn.__doc__ = description
    fn.__dict__["__tool_definition__"] = ToolDefinition(
        fn=fn,
        schema=schema,
        policy=policy if policy is not None else ToolRuntimePolicy(),
        preview=preview,
    )
    return fn


def _calling(handler: ToolHandler) -> Callable[..., object]:
    """A function of keyword arguments that hands them to ``handler`` in one dict.

    A coroutine function for an ``async`` handler, so the executor awaits it on the loop; a plain
    one otherwise, which the executor runs in a thread. No ``__wrapped__``: the executor reads
    the signature, ``(**arguments)``, not the handler's.
    """
    if inspect.iscoroutinefunction(handler):

        async def call_async(**arguments: object) -> object:
            return await handler(dict(arguments))

        return call_async

    def call(**arguments: object) -> object:
        return handler(dict(arguments))

    return call


def _standalone(name: str, input_schema: Mapping[str, object]) -> dict[str, object]:
    """A copy of the schema in plain JSON values, with its local references inlined.

    Raises:
        ValueError: As :func:`tool_from_schema` says.
    """
    try:
        schema = json.loads(json.dumps(dict(input_schema), allow_nan=False))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"input_schema of tool {name!r} is not JSON: {exc}") from exc
    try:
        _check_nodes(schema)
        schema = _inline_local_refs(schema)
    except ValueError as exc:
        raise ValueError(f"input_schema of tool {name!r}: {exc}") from exc
    except RecursionError as exc:  # the walk is bounded; a caller deep in its own stack is not
        raise ValueError(f"input_schema of tool {name!r} nests too deep to walk") from exc
    if schema.get("type") != "object":
        raise ValueError(
            f'input_schema of tool {name!r} must describe an object: its root needs "type": '
            f'"object", got {schema.get("type")!r}'
        )
    return schema


def _check_nodes(schema: object) -> None:
    """Refuse a reference outside the schema, and nesting the validator could not walk.

    The walk has no recursion of its own, since the schema may nest deep.

    Raises:
        ValueError: A ``$ref`` to a file or the network, or a node deeper than the limit.
    """
    for node, depth in _nodes(schema):
        if depth > _SCHEMA_DEPTH_LIMIT:
            raise ValueError(f"the schema nests deeper than {_SCHEMA_DEPTH_LIMIT} levels")
        ref = node.get("$ref") if isinstance(node, dict) else None
        if isinstance(ref, str) and not ref.startswith("#"):
            raise ValueError(
                f"the schema refers to {ref!r}, outside itself: only local references "
                "('#...') are followed, never one to a file or the network"
            )


def _nodes(schema: object) -> Iterator[tuple[object, int]]:
    """Every node of the schema, with how deep it sits."""
    pending: list[tuple[object, int]] = [(schema, 0)]
    while pending:
        node, depth = pending.pop()
        yield node, depth
        if isinstance(node, dict):
            pending.extend((child, depth + 1) for child in node.values())
        elif isinstance(node, list):
            pending.extend((child, depth + 1) for child in node)
