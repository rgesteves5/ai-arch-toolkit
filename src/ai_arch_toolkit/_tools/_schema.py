"""Schema inference — derive JSON Schema tool definitions from Python functions."""

from __future__ import annotations

import dataclasses
import enum
import inspect
import types
import typing
from collections.abc import Callable
from typing import Any, get_type_hints

_PYTHON_TYPE_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _hint_to_json_schema(hint: Any) -> tuple[dict[str, object], bool]:
    """Convert a Python type hint to a JSON Schema dict.

    Returns (schema_dict, is_optional).
    """
    origin = typing.get_origin(hint)

    # Handle X | None (types.UnionType) and typing.Optional[X] (typing.Union)
    if origin is types.UnionType or origin is typing.Union:
        args = typing.get_args(hint)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            schema, _ = _hint_to_json_schema(non_none[0])
            return schema, True
        # Multi-type union (not just Optional) — fall back to string
        return {"type": "string"}, True

    # Handle Literal["a", "b"] or Literal[1, 2]
    if origin is typing.Literal:
        args = typing.get_args(hint)
        if args and all(isinstance(a, int) for a in args):
            return {"type": "integer", "enum": list(args)}, False
        return {"type": "string", "enum": list(args)}, False

    # Handle enum.Enum subclasses
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        values = [m.value for m in hint]
        if values and all(isinstance(v, int) for v in values):
            return {"type": "integer", "enum": values}, False
        return {"type": "string", "enum": values}, False

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
        return hint.model_json_schema(), False

    # Primitive types
    json_type = _PYTHON_TYPE_TO_JSON.get(hint)
    if json_type is not None:
        return {"type": json_type}, False

    # Unknown — fallback to string
    return {"type": "string"}, False


def _is_typeddict(hint: Any) -> bool:
    """Check if a type is a TypedDict."""
    return isinstance(hint, type) and hasattr(hint, "__required_keys__")


def _typeddict_to_schema(hint: Any) -> dict[str, object]:
    """Convert a TypedDict to JSON Schema."""
    try:
        hints = get_type_hints(hint)
    except Exception:
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
    except Exception:
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

    Returns ``{"name": ..., "description": ..., "input_schema": {...}}``.
    """
    tool_name = name or fn.__name__
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = getattr(fn, "__annotations__", {})

    sig = inspect.signature(fn)
    param_descriptions = _parse_param_descriptions(fn)

    properties: dict[str, object] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        hint = hints.get(param_name)
        if hint is not None:
            try:
                schema, is_optional = _hint_to_json_schema(hint)
            except Exception:
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

    input_schema: dict[str, object] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    return {
        "name": tool_name,
        "description": _get_summary(fn),
        "input_schema": input_schema,
    }
