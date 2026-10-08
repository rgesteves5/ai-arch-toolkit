"""Validate and coerce tool-call arguments against a tool's input schema and signature.

Models — local ones especially — often send ``"3"`` for an integer or ``"true"`` for a boolean.
Arguments are checked before any gate runs, so an approval handler sees the values that will
actually run and an invalid call never reaches a human:

* ``integer`` accepts ints, integral floats, and integer strings (``"3"``, ``"3.0"``); never bools;
* ``number`` accepts ints, finite floats, and numeric strings; never bools;
* ``boolean`` accepts bools and the strings ``"true"`` / ``"false"`` (any case);
* ``enum`` is checked after coercion, and so are a parameter's top-level ``minimum``/``maximum``
  (from a ``Range`` in the signature or a ``schema=`` override), whose refusal names the range;
* ``anyOf``, ``oneOf`` and a ``type`` list (``["integer", "null"]``) keep a value that already
  matches a branch, and otherwise take the first branch that coerces it (``int | str`` keeps
  ``"1"`` as a string);
* ``string``, ``array``, ``object`` and untyped schemas (``Any``) are left as they are — the schema
  generator maps unknown Python types to ``string``, so rejecting non-strings would refuse valid
  calls — and so is everything nested in an array or an object (D7);
* ``None`` passes only where the parameter admits it — a ``None`` default, an annotation that
  includes ``None``, or no usable annotation (``Any``, untyped). The schema of a ``@tool`` does not
  record this, so it is read from the signature; a keyword that arrives through ``**kwargs`` (every
  argument of a ``tool_from_schema`` tool) admits it where its schema does (untyped, or ``null``
  among its types or branches). Elsewhere ``null`` is refused like any other wrong type;
* a parameter the schema does not require but the function has no default for
  (``query: str | None``) receives ``None`` when the model omits it.

Required arguments must be present, and arguments the schema does not declare are refused unless
the function takes ``**kwargs`` and the schema's root leaves them open (``additionalProperties:
false`` refuses them, D62). Finally the arguments must bind to the function's signature.
"""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, get_type_hints

from ai_arch_toolkit.core._tools._schema import _hint_to_json_schema

_INTEGER_TEXT = re.compile(r"^[+-]?\d+$")
_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


class ArgumentError(Exception):
    """Tool-call arguments that do not fit the tool's schema or signature."""

    def __init__(self, message: str, argument: str | None = None) -> None:
        super().__init__(message)
        self.argument = argument


def validate_arguments(
    fn: Callable[..., Any], schema: Mapping[str, Any], arguments: Mapping[str, Any]
) -> dict[str, Any]:
    """Return ``arguments`` coerced to ``schema``, or raise :class:`ArgumentError`."""
    if not isinstance(arguments, Mapping):  # e.g. a custom gate's GateModify with a list
        raise ArgumentError(f"arguments must be a mapping, got {type(arguments).__name__}")
    coerced = dict(arguments)
    properties = _declared(schema)
    if properties is None:
        return coerced
    _check_required(schema.get("required"), coerced)
    _check_declared(fn, schema, properties, coerced)

    known = _none_allowed(fn)
    for name, value in arguments.items():
        declared = properties.get(name)
        if isinstance(declared, Mapping):
            coerced[name] = _checked(name, value, declared, _admits_none(known, name, declared))

    for param in _parameters(fn):
        if param.name in properties and param.name not in coerced and _needs_value(param):
            coerced[param.name] = None  # optional in the schema, required by the signature
    return coerced


def _declared(schema: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The arguments the root declares: its ``properties``, none at all for a closed root
    without them, ``None`` for a schema that says nothing about its arguments."""
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        return properties
    return {} if _closed(schema) else None


def _closed(schema: Mapping[str, Any]) -> bool:
    return schema.get("additionalProperties") is False


def _check_required(required: object, arguments: Mapping[str, Any]) -> None:
    if isinstance(required, Sequence) and not isinstance(required, str):
        missing = [name for name in required if name not in arguments]
        if missing:
            raise ArgumentError(f"is missing required argument(s) {_names(missing)}", missing[0])


def _check_declared(
    fn: Callable[..., Any],
    schema: Mapping[str, Any],
    properties: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> None:
    """Refuse an argument the schema does not declare, unless ``**kwargs`` takes it and the
    root leaves it open."""
    if not _closed(schema) and _accepts_extra_keywords(fn):
        return
    unexpected = [name for name in arguments if name not in properties]
    if unexpected:
        expected = ", ".join(properties) or "none"
        raise ArgumentError(
            f"got unexpected argument(s) {_names(unexpected)}; expected: {expected}",
            unexpected[0],
        )


def _checked(name: str, value: Any, declared: Mapping[str, Any], admits_none: bool) -> Any:
    """``value`` coerced to its declared schema and inside its bounds, else ``ArgumentError``."""
    if value is None and not admits_none:
        raise ArgumentError(f"argument {name!r}: expected {_describe(declared)}, got null", name)
    ok, coerced, expected = _coerce(value, declared)
    if ok and not _within(coerced, declared):
        ok, expected = False, _describe_bounds(declared)
    if not ok:
        got = f"{type(value).__name__} {_short(value)}"
        raise ArgumentError(f"argument {name!r}: expected {expected}, got {got}", name)
    return coerced


def _within(value: Any, schema: Mapping[str, Any]) -> bool:
    """Whether a number is inside the schema's ``minimum``/``maximum``; other values always are."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return True
    low, high = schema.get("minimum"), schema.get("maximum")
    too_low = isinstance(low, int | float) and value < low
    too_high = isinstance(high, int | float) and value > high
    return not (too_low or too_high)


def _describe_bounds(schema: Mapping[str, Any]) -> str:
    """``integer from 1 to 25``, ``number at least 0``, ``integer at most 100``."""
    branches = _branches(schema)
    kinds = [b.get("type") for b in branches or [schema] if b.get("type") in ("integer", "number")]
    kind = kinds[0] if kinds else "number"
    low, high = schema.get("minimum"), schema.get("maximum")
    if low is not None and high is not None:
        return f"{kind} from {low} to {high}"
    if low is not None:
        return f"{kind} at least {low}"
    return f"{kind} at most {high}"


def _none_allowed(fn: Callable[..., Any]) -> dict[str, bool] | None:
    """Per named parameter, whether ``None`` is a value its default or annotation admits.

    ``None`` when the signature or its hints cannot be read: every argument then admits ``None``,
    since a valid call is never refused because the answer could not be worked out. A parameter
    without a usable annotation admits it too.
    """
    try:
        parameters = inspect.signature(fn).parameters
        hints = get_type_hints(fn)
    except Exception:
        return None
    allowed: dict[str, bool] = {}
    for name, param in parameters.items():
        if param.kind in _VARIADIC:
            continue
        if param.default is None or name not in hints:
            allowed[name] = True
            continue
        schema, is_optional = _hint_to_json_schema(hints[name])
        allowed[name] = is_optional or not schema  # the empty schema is ``Any`` / ``object``
    return allowed


def _admits_none(known: Mapping[str, bool] | None, name: str, declared: Mapping[str, Any]) -> bool:
    """Whether ``null`` is a value for the argument ``name``: its parameter says, else (a keyword
    that arrives through ``**kwargs``) its schema does (D62)."""
    if known is None:
        return True
    if name in known:
        return known[name]
    return _admits_null(declared)


def _admits_null(schema: Mapping[str, Any]) -> bool:
    """Whether a schema lets ``null`` through: untyped, typed ``null``, or a branch that does."""
    branches = _branches(schema)
    if branches:
        return any(_admits_null(branch) for branch in branches)
    allowed = _enum(schema)
    return schema.get("type") in (None, "null") and (allowed is None or None in allowed)


def _parameters(fn: Callable[..., Any]) -> list[inspect.Parameter]:
    try:
        return list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return []


def _needs_value(param: inspect.Parameter) -> bool:
    return param.default is inspect.Parameter.empty and param.kind not in _VARIADIC


def bind_arguments(
    fn: Callable[..., Any], arguments: Mapping[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Split named arguments into a call and check it binds, before the tool runs.

    Positional-only parameters cannot be passed by name, so their values go by position, in
    signature order (``inspect.signature`` follows ``__wrapped__``, so a ``@tool`` wrapper reports
    the decorated function's parameters); an omitted positional-only parameter that precedes a
    supplied one takes its default. Checking the binding first keeps a ``TypeError`` raised inside
    the tool's own body distinct from arguments that never matched its signature.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # no introspectable signature: pass everything by name
        return [], dict(arguments)

    keywords = dict(arguments)
    positional: list[Any] = []
    positional_only = [
        p for p in signature.parameters.values() if p.kind is inspect.Parameter.POSITIONAL_ONLY
    ]
    supplied = [index for index, p in enumerate(positional_only) if p.name in keywords]
    if supplied:
        for param in positional_only[: supplied[-1] + 1]:
            if param.name in keywords:
                positional.append(keywords.pop(param.name))
            elif param.default is not inspect.Parameter.empty:
                positional.append(param.default)
            else:
                break  # a required one is missing: binding reports it below

    try:
        signature.bind(*positional, **keywords)
    except TypeError as exc:
        raise ArgumentError(f"argument mismatch: {exc}") from exc
    return positional, keywords


def _accepts_extra_keywords(fn: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return True  # can't tell: let the binding check decide
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)


def _branches(schema: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The alternatives a schema offers: its ``anyOf`` or ``oneOf`` branches, or one branch per
    member of a ``type`` list, each with the schema's other keywords (D62).

    A ``oneOf`` is read as an ``anyOf``: a value that fits two branches is the server's to refuse.
    """
    for keyword in ("anyOf", "oneOf"):
        branches = [branch for branch in schema.get(keyword) or () if isinstance(branch, Mapping)]
        if branches:
            return branches
    kinds = schema.get("type")
    if not isinstance(kinds, list):
        return []
    rest = {key: value for key, value in schema.items() if key != "type"}
    return [{**rest, "type": kind} for kind in kinds]


def _coerce(value: Any, schema: Mapping[str, Any]) -> tuple[bool, Any, str]:
    """``(ok, coerced value, description of what was expected)``."""
    if value is None:
        return True, None, ""
    branches = _branches(schema)
    if branches:
        return _coerce_to_a_branch(value, branches)
    kind = schema.get("type")
    convert = _CONVERTERS.get(kind, _as_it_is) if isinstance(kind, str) else _as_it_is
    ok, coerced = convert(value)
    allowed = _enum(schema)
    if ok and allowed is not None and coerced not in allowed:
        return False, value, f"one of {list(allowed)!r}"
    return ok, coerced, _describe(schema)


def _coerce_to_a_branch(value: Any, branches: list[Mapping[str, Any]]) -> tuple[bool, Any, str]:
    """A value that already matches a branch as it is, else the first branch that coerces it."""
    if any(_matches(value, branch) for branch in branches):
        return True, value, ""
    for branch in branches:
        ok, coerced, _ = _coerce(value, branch)
        if ok:
            return True, coerced, ""
    return False, value, " or ".join(_describe(branch) for branch in branches)


def _matches(value: Any, schema: Mapping[str, Any]) -> bool:
    """Whether ``value`` already fits ``schema`` exactly, with no coercion."""
    branches = _branches(schema)
    if branches:
        return any(_matches(value, branch) for branch in branches)
    kind = schema.get("type")
    if kind == "integer":
        fits = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "number":
        fits = isinstance(value, int | float) and not isinstance(value, bool)
    elif kind == "boolean":
        fits = isinstance(value, bool)
    elif kind == "string":
        fits = isinstance(value, str)
    elif kind == "array":
        fits = isinstance(value, list)
    elif kind == "object":
        fits = isinstance(value, dict)
    elif kind == "null":
        fits = value is None
    else:
        fits = True
    allowed = _enum(schema)
    if fits and allowed is not None:
        fits = value in allowed
    return fits


def _enum(schema: Mapping[str, Any]) -> Sequence[Any] | None:
    allowed = schema.get("enum")
    if isinstance(allowed, Sequence) and not isinstance(allowed, str):
        return allowed
    return None


def _to_integer(value: Any) -> tuple[bool, Any]:
    if isinstance(value, bool):
        return False, value
    if isinstance(value, int):
        return True, value
    if isinstance(value, float):
        return (True, int(value)) if value.is_integer() else (False, value)
    if isinstance(value, str):
        text = value.strip()
        if _INTEGER_TEXT.match(text):
            number = _parse_int(text)
            return (True, number) if number is not None else (False, value)
        number = _parse_float(text)
        if number is not None and number.is_integer():
            return True, int(number)
    return False, value


def _to_number(value: Any) -> tuple[bool, Any]:
    if isinstance(value, bool):
        return False, value
    if isinstance(value, int):
        return True, value
    if isinstance(value, float):
        return math.isfinite(value), value
    if isinstance(value, str):
        text = value.strip()
        if _INTEGER_TEXT.match(text):
            integer = _parse_int(text)
            return (True, integer) if integer is not None else (False, value)
        number = _parse_float(text)
        if number is not None:
            return True, number
    return False, value


def _to_boolean(value: Any) -> tuple[bool, Any]:
    if isinstance(value, bool):
        return True, value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return True, value.strip().lower() == "true"
    return False, value


def _to_null(value: Any) -> tuple[bool, Any]:
    return False, value  # ``None`` itself never gets here


def _as_it_is(value: Any) -> tuple[bool, Any]:
    return True, value


_CONVERTERS: dict[str, Callable[[Any], tuple[bool, Any]]] = {
    "integer": _to_integer,
    "number": _to_number,
    "boolean": _to_boolean,
    "null": _to_null,
}


def _parse_int(text: str) -> int | None:
    try:
        return int(text)
    except ValueError:  # longer than Python's integer string conversion limit
        return None


def _parse_float(text: str) -> float | None:
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _describe(schema: Mapping[str, Any]) -> str:
    branches = _branches(schema)
    if branches:
        return " or ".join(_describe(branch) for branch in branches)
    kind = schema.get("type")
    return kind if isinstance(kind, str) else "any value"


def _names(names: Sequence[str]) -> str:
    return ", ".join(repr(name) for name in names)


def _short(value: Any, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
