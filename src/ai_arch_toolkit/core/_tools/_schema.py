"""Schema inference — derive JSON Schema tool definitions from Python functions."""

from __future__ import annotations

import dataclasses
import enum
import functools
import inspect
import logging
import types
import typing
from collections.abc import Callable
from typing import Any, get_type_hints

from ai_arch_toolkit.core._tools._definition import ToolSchema

logger = logging.getLogger(__name__)

_PYTHON_TYPE_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _hint_to_json_schema(hint: Any) -> tuple[dict[str, object], bool]:
    """Convert a Python type hint to a JSON Schema dict.

    A union of several non-``None`` types becomes ``anyOf`` (never a ``type`` list, which
    Gemini rejects); members whose schemas are identical collapse into one. ``Any`` and
    ``object`` become the empty schema (any JSON value); a type the generator does not know
    becomes ``string``, as does a parameter with no annotation.

    Returns:
        ``(schema, is_optional)``, where ``is_optional`` is true only when ``None`` is a
        member of a union.
    """
    # PEP 695 aliases (``type Count = int``) describe their value.
    if isinstance(hint, typing.TypeAliasType):
        return _hint_to_json_schema(hint.__value__)

    # ``Any`` and ``object`` accept every JSON value, so the schema sets no constraint.
    if hint is typing.Any or hint is object:
        return {}, False

    origin = typing.get_origin(hint)

    # Handle unions: X | Y (types.UnionType) and typing.Union / typing.Optional
    if origin is types.UnionType or origin is typing.Union:
        args = typing.get_args(hint)
        variants: list[dict[str, object]] = []
        for arg in args:
            if arg is type(None):
                continue
            variant, _ = _hint_to_json_schema(arg)
            if variant not in variants:
                variants.append(variant)
        is_optional = type(None) in args
        if len(variants) == 1:
            return variants[0], is_optional
        return {"anyOf": variants}, is_optional

    # Handle Literal["a", "b"] or Literal[1, 2]
    if origin is typing.Literal:
        return _enum_schema(list(typing.get_args(hint))), False

    # Handle enum.Enum subclasses
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return _enum_schema([m.value for m in hint]), False

    # Handle list / list[T]
    if origin is list:
        args = typing.get_args(hint)
        if args:
            inner_schema, _ = _hint_to_json_schema(args[0])
            return {"type": "array", "items": inner_schema}, False
        return {"type": "array"}, False
    if hint is list:
        return {"type": "array"}, False

    # Handle tuple[str, int] (fixed) and tuple[str, ...] (variable)
    if origin is tuple:
        args = typing.get_args(hint)
        if args:
            if len(args) == 2 and args[1] is Ellipsis:
                inner_schema, _ = _hint_to_json_schema(args[0])
                return {"type": "array", "items": inner_schema}, False
            prefix_items = [_hint_to_json_schema(a)[0] for a in args]
            return {
                "type": "array",
                "prefixItems": prefix_items,
            }, False
        return {"type": "array"}, False

    # Handle dict / dict[K, V]
    if origin is dict or hint is dict:
        return {"type": "object"}, False

    # Handle dataclasses
    if dataclasses.is_dataclass(hint) and isinstance(hint, type):
        return _dataclass_to_schema(hint), False

    # Handle TypedDict
    if _is_typeddict(hint):
        return _typeddict_to_schema(hint), False

    # Handle Pydantic BaseModel (duck-typed, no import)
    if isinstance(hint, type) and hasattr(hint, "model_json_schema"):
        return _inline_local_refs(hint.model_json_schema()), False

    # Primitive types
    if isinstance(hint, type):
        json_type = _PYTHON_TYPE_TO_JSON.get(hint)
        if json_type is not None:
            return {"type": json_type}, False

    # Unknown — fallback to string
    return {"type": "string"}, False


def _enum_schema(values: list[Any]) -> dict[str, object]:
    """Schema for a fixed set of values, typed by what they are (``bool`` is not an integer)."""
    if not values:
        return {"type": "string"}
    if all(isinstance(v, bool) for v in values):
        if set(values) == {True, False}:
            return {"type": "boolean"}
        return {"type": "boolean", "enum": values}
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return {"type": "integer", "enum": values}
    return {"type": "string", "enum": values}


_LOCAL_DEFINITION = "#/$defs/"


def _inline_local_refs(schema: dict[str, Any]) -> dict[str, object]:
    """Resolve ``#/$defs/...`` references so the schema stands on its own.

    Pydantic describes nested models as ``$ref`` pointers into a ``$defs`` table at the top of
    the model's schema. Embedded as one tool parameter, that table no longer sits at the root
    the pointers name, so a provider sees dangling references (Gemini rejects the tool). A
    reference to a definition being expanded (a recursive model) is kept, together with the
    table, which :func:`infer_schema` hoists to the tool's root.
    """
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        return schema

    def resolve(node: Any, expanding: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [resolve(item, expanding) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith(_LOCAL_DEFINITION):
            name = ref.removeprefix(_LOCAL_DEFINITION)
            if name in definitions and name not in expanding:
                target = resolve(definitions[name], expanding | {name})
                siblings = {k: resolve(v, expanding) for k, v in node.items() if k != "$ref"}
                return {**target, **siblings}  # a field's description sits next to its $ref
        return {key: resolve(value, expanding) for key, value in node.items()}

    inlined: dict[str, object] = resolve(
        {key: value for key, value in schema.items() if key != "$defs"}, frozenset()
    )
    if _mentions_local_ref(inlined):
        inlined["$defs"] = definitions
    return inlined


def _mentions_local_ref(node: Any) -> bool:
    if isinstance(node, list):
        return any(_mentions_local_ref(item) for item in node)
    if not isinstance(node, dict):
        return False
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith(_LOCAL_DEFINITION):
        return True
    return any(_mentions_local_ref(value) for value in node.values())


def _hoist_definitions(schema: dict[str, object]) -> dict[str, object]:
    """Move every nested ``$defs`` table to the schema's root, where ``#/$defs/`` points."""
    hoisted: dict[str, object] = {}

    def strip(node: Any) -> Any:
        if isinstance(node, list):
            return [strip(item) for item in node]
        if not isinstance(node, dict):
            return node
        table = node.get("$defs")
        if isinstance(table, dict):
            for name, definition in table.items():
                if name in hoisted and hoisted[name] != definition:
                    logger.warning("Tool schema defines %r twice; keeping the first", name)
                    continue
                hoisted[name] = definition
        return {key: strip(value) for key, value in node.items() if key != "$defs"}

    stripped: dict[str, object] = strip(schema)
    if hoisted:
        stripped["$defs"] = hoisted
    return stripped


def _is_typeddict(hint: Any) -> bool:
    """Check if a type is a TypedDict."""
    return isinstance(hint, type) and hasattr(hint, "__required_keys__")


def _typeddict_to_schema(hint: Any) -> dict[str, object]:
    """Convert a TypedDict to JSON Schema."""
    try:
        hints = get_type_hints(hint)
    except (NameError, AttributeError, TypeError):
        logger.warning("Could not resolve type hints for TypedDict %s", hint.__name__)
        hints = {}
    properties: dict[str, object] = {}
    for name, h in hints.items():
        schema, _ = _hint_to_json_schema(h)
        properties[name] = schema
    required = list(hint.__required_keys__)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _dataclass_to_schema(hint: Any) -> dict[str, object]:
    """Convert a dataclass to JSON Schema."""
    try:
        hints = get_type_hints(hint)
    except (NameError, AttributeError, TypeError):
        logger.warning("Could not resolve type hints for dataclass %s", hint.__name__)
        hints = {}
    fields = dataclasses.fields(hint)
    properties: dict[str, object] = {}
    required: list[str] = []
    for f in fields:
        h = hints.get(f.name)
        if h is not None:
            schema, is_optional = _hint_to_json_schema(h)
        else:
            schema, is_optional = {"type": "string"}, False
        properties[f.name] = schema
        has_default = (
            f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
        )
        if not has_default and not is_optional:
            required.append(f.name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _parse_param_descriptions(fn: Callable[..., Any]) -> dict[str, str]:
    """Parse Google-style docstring Args section into param->description map."""
    doc = inspect.getdoc(fn)
    if not doc:
        return {}

    lines = doc.split("\n")
    result: dict[str, str] = {}
    in_args = False
    current_param: str | None = None
    current_desc: list[str] = []
    args_indent: int | None = None

    for line in lines:
        stripped = line.strip()

        # Detect start of Args section
        if stripped in ("Args:", "Arguments:", "Parameters:"):
            in_args = True
            args_indent = None
            continue

        # Detect end of Args section (another section header)
        if in_args and stripped and stripped.endswith(":") and not stripped.startswith(" "):
            section_name = stripped[:-1].strip()
            if section_name in (
                "Returns",
                "Raises",
                "Yields",
                "Examples",
                "Notes",
                "References",
                "Attributes",
                "See Also",
            ):
                if current_param:
                    result[current_param] = " ".join(current_desc).strip()
                in_args = False
                continue

        if not in_args:
            continue

        if not stripped:
            continue

        indent = len(line) - len(line.lstrip())
        if args_indent is None:
            args_indent = indent

        if indent == args_indent:
            if current_param:
                result[current_param] = " ".join(current_desc).strip()
            if ":" in stripped:
                param_part, _, desc_part = stripped.partition(":")
                param_name = param_part.split("(")[0].strip()
                current_param = param_name
                current_desc = [desc_part.strip()] if desc_part.strip() else []
            else:
                current_param = None
                current_desc = []
        elif indent > (args_indent or 0) and current_param:
            current_desc.append(stripped)

    if in_args and current_param:
        result[current_param] = " ".join(current_desc).strip()

    return result


def _get_summary(fn: Callable[..., Any]) -> str:
    """Extract docstring summary (text before Args section)."""
    doc = inspect.getdoc(fn)
    if not doc:
        return ""
    lines = doc.split("\n")
    summary_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped in ("Args:", "Arguments:", "Parameters:"):
            break
        summary_lines.append(line)
    while summary_lines and not summary_lines[-1].strip():
        summary_lines.pop()
    return "\n".join(summary_lines).strip()


def _described_function(fn: Callable[..., Any]) -> Callable[..., Any]:
    """The function whose name, type hints and docstring describe ``fn``.

    A ``functools.partial`` has none of its own; its signature (from ``inspect.signature``)
    already drops the arguments it binds.
    """
    return fn.func if isinstance(fn, functools.partial) else fn


def callable_name(fn: Callable[..., Any]) -> str:
    """The tool name of an undecorated callable: its function's name, else its type's."""
    return getattr(_described_function(fn), "__name__", None) or type(fn).__name__


def _is_json_serializable(value: Any) -> bool:
    """Check if a value can be included as a JSON Schema default."""
    return isinstance(value, (str, int, float, bool, type(None), list, dict))


def infer_schema(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    overrides: dict[str, dict[str, object]] | None = None,
) -> dict[str, Any]:
    """Build a tool definition dict from a function's type hints and docstring.

    Variadic ``*args`` / ``**kwargs`` parameters are never part of the schema.

    Returns ``{"name": ..., "description": ..., "input_schema": {...}}``.
    """
    described = _described_function(fn)
    tool_name = name or callable_name(fn)
    try:
        hints = get_type_hints(described)
    except (NameError, AttributeError, TypeError):
        logger.warning("Could not resolve type hints for %s, using annotations", tool_name)
        hints = getattr(described, "__annotations__", {})

    sig = inspect.signature(fn)
    param_descriptions = _parse_param_descriptions(described)

    properties: dict[str, object] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        # Arguments arrive by name, which can never bind a variadic parameter itself.
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        hint = hints.get(param_name)
        if hint is not None:
            try:
                schema, is_optional = _hint_to_json_schema(hint)
            except (NameError, AttributeError, TypeError):
                logger.warning("Could not convert hint for parameter %r", param_name)
                schema, is_optional = {"type": "string"}, False
        else:
            schema, is_optional = {"type": "string"}, False

        desc = param_descriptions.get(param_name)
        if desc:
            schema = {**schema, "description": desc}

        # Include default value in schema when JSON-serializable
        if param.default is not inspect.Parameter.empty:
            if _is_json_serializable(param.default):
                schema = {**schema, "default": param.default}
        elif not is_optional:
            required.append(param_name)

        properties[param_name] = schema

    # Apply overrides
    if overrides:
        for pname, override in overrides.items():
            if pname in properties:
                properties[pname] = {
                    **properties[pname],  # type: ignore[arg-type]
                    **override,
                }
            else:
                properties[pname] = override

    input_schema = _hoist_definitions(
        {"type": "object", "properties": properties, "required": required}
    )

    return {
        "name": tool_name,
        "description": _get_summary(described),
        "input_schema": input_schema,
    }


def tool_schema(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    overrides: dict[str, dict[str, object]] | None = None,
) -> ToolSchema:
    """Build a provider-facing ``ToolSchema`` from a function."""
    d = infer_schema(fn, name=name, overrides=overrides)
    return ToolSchema(
        name=d["name"],
        description=d["description"],
        input_schema=d["input_schema"],
    )
