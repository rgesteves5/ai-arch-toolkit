"""Schema inference — derive JSON Schema tool definitions from Python functions."""

from __future__ import annotations

import dataclasses
import enum
import functools
import inspect
import logging
import math
import types
import typing
from collections.abc import Callable, Mapping
from typing import Any, get_type_hints

from ai_arch_toolkit.core._tools._definition import ToolSchema

logger = logging.getLogger(__name__)

_PYTHON_TYPE_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}
_NUMERIC_TYPES = ("integer", "number")


@dataclasses.dataclass(frozen=True, slots=True)
class Range:
    """Inclusive bounds for a numeric tool parameter.

    Put it in the parameter's annotation, ``max_results: Annotated[int, Range(1, 25)] = 10``: the
    schema the model reads carries the bounds as ``minimum`` and ``maximum``, and the executor
    refuses a value outside them with a ``validation_error`` that names the range, so the tool
    never has to adjust a value without saying so. It works on ``int``, ``float`` and unions with a
    numeric member (``Annotated[int | None, Range(1, 25)]``).

    Attributes:
        minimum: The smallest value allowed, or ``None`` for no lower bound.
        maximum: The largest value allowed, or ``None`` for no upper bound.

    Raises:
        ValueError: When there is no bound, a bound is not a finite number (``bool`` included),
            or ``minimum`` is above ``maximum``.
    """

    minimum: int | float | None = None
    maximum: int | float | None = None

    def __post_init__(self) -> None:
        bounds = [bound for bound in (self.minimum, self.maximum) if bound is not None]
        if not bounds:
            raise ValueError("Range needs a minimum, a maximum or both")
        if not all(_finite_number(bound) for bound in bounds):
            raise ValueError(f"Range bounds must be finite numbers, got {self!r}")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"Range minimum {self.minimum} is above its maximum {self.maximum}")

    def schema(self) -> dict[str, object]:
        """The JSON Schema keywords for these bounds."""
        keywords: dict[str, object] = {}
        if self.minimum is not None:
            keywords["minimum"] = self.minimum
        if self.maximum is not None:
            keywords["maximum"] = self.maximum
        return keywords


def _finite_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


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
    hint = _bare(hint)

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

# Inlining can multiply a schema: a definition that refers twice to the next one doubles at each
# level (1.9 KB with 18 such levels became 17.5 MB in 1.4 s). A schema that arrives at run time,
# from an MCP server, must fail fast instead. The size is in about one unit per character of
# JSON; the depth matches the toolkit's other loaded data (D59).
_INLINE_SIZE_LIMIT = 1_000_000
_SCHEMA_DEPTH_LIMIT = 100


def _inline_local_refs(schema: dict[str, Any]) -> dict[str, object]:
    """Resolve ``#/$defs/...`` references so the schema stands on its own.

    Pydantic describes nested models as ``$ref`` pointers into a ``$defs`` table at the top of
    the model's schema. Embedded as one tool parameter, that table no longer sits at the root
    the pointers name, so a provider sees dangling references (Gemini rejects the tool). A
    reference to a definition being expanded (a recursive model) is kept, together with the
    table, which :func:`infer_schema` hoists to the tool's root.

    Raises:
        ValueError: The inlined schema would take more than about 1,000,000 characters of JSON
            or nest deeper than 100 levels. The walk stops there, so a hostile schema fails in
            milliseconds.
    """
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        return schema
    root = {key: value for key, value in schema.items() if key != "$defs"}
    inlined: dict[str, object] = _Inlining(definitions).resolve(root, frozenset(), 0)
    if _mentions_local_ref(inlined):
        inlined["$defs"] = definitions
    return inlined


class _Inlining:
    """One walk that inlines a schema's ``#/$defs/...`` references, and what it may still spend."""

    __slots__ = ("_definitions", "_left")

    def __init__(self, definitions: dict[str, Any]) -> None:
        self._definitions = definitions
        self._left = _INLINE_SIZE_LIMIT

    def resolve(self, node: Any, expanding: frozenset[str], depth: int) -> Any:
        """``node`` with its references inlined, but those to a definition being expanded."""
        self._spend(node, depth)
        if isinstance(node, list):
            return [self.resolve(item, expanding, depth + 1) for item in node]
        if not isinstance(node, dict):
            return node
        name = self._target(node, expanding)
        if name is None:
            return {key: self.resolve(value, expanding, depth + 1) for key, value in node.items()}
        target = self.resolve(self._definitions[name], expanding | {name}, depth)
        siblings = {
            key: self.resolve(value, expanding, depth + 1)
            for key, value in node.items()
            if key != "$ref"
        }
        return {**target, **siblings}  # a field's description sits next to its $ref

    def _target(self, node: dict[str, Any], expanding: frozenset[str]) -> str | None:
        """The definition ``node`` refers to, when it is one to inline."""
        ref = node.get("$ref")
        if not isinstance(ref, str) or not ref.startswith(_LOCAL_DEFINITION):
            return None
        name = ref.removeprefix(_LOCAL_DEFINITION)
        return name if name in self._definitions and name not in expanding else None

    def _spend(self, node: object, depth: int) -> None:
        self._left -= _own_size(node)
        if self._left < 0:
            raise ValueError(
                f"the schema would take more than about {_INLINE_SIZE_LIMIT:,} characters once "
                "its $ref references are inlined"
            )
        if depth > _SCHEMA_DEPTH_LIMIT:
            raise ValueError(
                f"the schema would nest deeper than {_SCHEMA_DEPTH_LIMIT} levels once its $ref "
                "references are inlined"
            )


def _own_size(node: object) -> int:
    """About the characters of JSON a node takes, its children's apart."""
    if isinstance(node, dict):
        return 2 + sum(len(str(key)) + 4 for key in node)
    if isinstance(node, list):
        return 2 + len(node)
    if isinstance(node, str):
        return len(node) + 2
    return 4


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

    Variadic ``*args`` / ``**kwargs`` parameters are never part of the schema. A :class:`Range`
    in a parameter's ``Annotated`` hint adds its bounds.

    Returns ``{"name": ..., "description": ..., "input_schema": {...}}``.

    Raises:
        ValueError: A :class:`Range` on a parameter whose type has no numbers.
    """
    described = _described_function(fn)
    tool_name = name or callable_name(fn)
    hints = _type_hints(described, tool_name)
    annotated = _annotated_hints(described)
    descriptions = _parse_param_descriptions(described)

    properties: dict[str, object] = {}
    required: list[str] = []
    for param_name, param in inspect.signature(fn).parameters.items():
        if not _in_schema(param):
            continue
        schema, needed = _parameter_schema(
            param, hints.get(param_name), _range_of(annotated.get(param_name))
        )
        if description := descriptions.get(param_name):
            schema = {**schema, "description": description}
        if param.default is not inspect.Parameter.empty and _is_json_serializable(param.default):
            schema = {**schema, "default": param.default}
        properties[param_name] = schema
        if needed:
            required.append(param_name)
    for param_name, override in (overrides or {}).items():
        current = properties.get(param_name)
        properties[param_name] = {**current, **override} if isinstance(current, dict) else override

    input_schema = _hoist_definitions(
        {"type": "object", "properties": properties, "required": required}
    )

    return {
        "name": tool_name,
        "description": _get_summary(described),
        "input_schema": input_schema,
    }


def _type_hints(described: Callable[..., Any], tool_name: str) -> dict[str, Any]:
    """The function's resolved hints, or its raw annotations when they do not resolve."""
    try:
        return get_type_hints(described)
    except (NameError, AttributeError, TypeError):
        logger.warning("Could not resolve type hints for %s, using annotations", tool_name)
        return getattr(described, "__annotations__", {})


def _annotated_hints(described: Callable[..., Any]) -> dict[str, Any]:
    """The hints with their ``Annotated`` metadata, where a :class:`Range` lives.

    The schema itself comes from the plain hints, which keep every nested type as before; these
    are read only for the bounds. Empty when the hints do not resolve.
    """
    try:
        return get_type_hints(described, include_extras=True)
    except (NameError, AttributeError, TypeError):
        return {}


def _in_schema(param: inspect.Parameter) -> bool:
    """Arguments arrive by name, which never binds ``self``/``cls`` or a variadic parameter."""
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    return param.name not in ("self", "cls") and param.kind not in variadic


def _parameter_schema(
    param: inspect.Parameter, hint: Any, bounds: Range | None
) -> tuple[dict[str, object], bool]:
    """``(schema, required)`` for one parameter: its hint's schema, bounded by ``bounds``."""
    schema: dict[str, object] = {"type": "string"}
    is_optional = False
    if hint is not None:
        try:
            schema, is_optional = _hint_to_json_schema(hint)
        except (NameError, AttributeError, TypeError):
            logger.warning("Could not convert hint for parameter %r", param.name)
    if bounds is not None:
        if not _numeric(schema):
            raise ValueError(f"Range on parameter {param.name!r} needs an int or float type")
        schema = {**schema, **bounds.schema()}
    return schema, param.default is inspect.Parameter.empty and not is_optional


def _range_of(hint: Any) -> Range | None:
    """The :class:`Range` of ``Annotated[..., Range(...)]``, also as a member of a union and
    behind a PEP 695 alias (``type Chars = Annotated[int, Range(1, 20)]``)."""
    hint = _unaliased(hint)
    members = [hint]
    if typing.get_origin(hint) in (types.UnionType, typing.Union):
        members = [_unaliased(member) for member in typing.get_args(hint)]
    for member in members:
        if typing.get_origin(member) is typing.Annotated:
            found = [item for item in typing.get_args(member)[1:] if isinstance(item, Range)]
            if found:
                return found[0]
    return None


def _bare(hint: Any) -> Any:
    """The type a hint describes: PEP 695 aliases (``type Count = int``) followed and
    ``Annotated`` metadata dropped. Inside an alias ``get_type_hints`` cannot drop it;
    ``_range_of`` reads its ``Range``."""
    while True:
        if isinstance(hint, typing.TypeAliasType):
            hint = hint.__value__
        elif typing.get_origin(hint) is typing.Annotated:
            hint = typing.get_args(hint)[0]
        else:
            return hint


def _unaliased(hint: Any) -> Any:
    """The value behind PEP 695 aliases, which may alias one another."""
    while isinstance(hint, typing.TypeAliasType):
        hint = hint.__value__
    return hint


def _numeric(schema: Mapping[str, object]) -> bool:
    """Whether the schema admits a number, itself or through an ``anyOf`` branch."""
    branches = schema.get("anyOf")
    candidates = branches if isinstance(branches, list) else [schema]
    return any(isinstance(c, dict) and c.get("type") in _NUMERIC_TYPES for c in candidates)


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
