"""Validate and coerce tool-call arguments against a tool's input schema and signature.

Models — local ones especially — often send ``"3"`` for an integer or ``"true"`` for a boolean.
Arguments are checked before any gate runs, so an approval handler sees the values that will
actually run and an invalid call never reaches a human:

* ``integer`` accepts ints, integral floats, and integer strings (``"3"``, ``"3.0"``); never bools;
* ``number`` accepts ints, finite floats, and numeric strings; never bools;
* ``boolean`` accepts bools and the strings ``"true"`` / ``"false"`` (any case);
* ``enum`` is checked after coercion;
* ``anyOf`` keeps a value that already matches a branch, and otherwise takes the first branch that
  coerces it (``int | str`` keeps ``"1"`` as a string);
* ``string``, ``array``, ``object`` and untyped schemas (``Any``) are left as they are — the schema
  generator maps unknown Python types to ``string``, so rejecting non-strings would refuse valid
  calls;
* ``None`` passes (the schema does not record whether a parameter is ``Optional``).

Required arguments must be present, and arguments the schema does not declare are refused unless
the function takes ``**kwargs``. Finally the arguments must bind to the function's signature.
"""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

_INTEGER_TEXT = re.compile(r"^[+-]?\d+$")


class ArgumentError(Exception):
    """Tool-call arguments that do not fit the tool's schema or signature."""

    def __init__(self, message: str, argument: str | None = None) -> None:
        super().__init__(message)
        self.argument = argument


def validate_arguments(
    fn: Callable[..., Any], schema: Mapping[str, Any], arguments: Mapping[str, Any]
) -> dict[str, Any]:
    """Return ``arguments`` coerced to ``schema``, or raise :class:`ArgumentError`."""
    coerced = dict(arguments)
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return coerced

    required = schema.get("required")
    if isinstance(required, Sequence) and not isinstance(required, str):
        missing = [name for name in required if name not in coerced]
        if missing:
            raise ArgumentError(f"is missing required argument(s) {_names(missing)}", missing[0])

    if not _accepts_extra_keywords(fn):
        unexpected = [name for name in coerced if name not in properties]
        if unexpected:
            expected = ", ".join(properties) or "none"
            raise ArgumentError(
                f"got unexpected argument(s) {_names(unexpected)}; expected: {expected}",
                unexpected[0],
            )

    for name, value in arguments.items():
        declared = properties.get(name)
        if not isinstance(declared, Mapping):
            continue
        ok, value_out, expected = _coerce(value, declared)
        if not ok:
            got = f"{type(value).__name__} {_short(value)}"
            raise ArgumentError(f"argument {name!r}: expected {expected}, got {got}", name)
        coerced[name] = value_out
    return coerced


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


def _coerce(value: Any, schema: Mapping[str, Any]) -> tuple[bool, Any, str]:
    """``(ok, coerced value, description of what was expected)``."""
    if value is None:
        return True, None, ""

    branches = [branch for branch in schema.get("anyOf") or () if isinstance(branch, Mapping)]
    if branches:
        if any(_matches(value, branch) for branch in branches):
            return True, value, ""
        for branch in branches:
            ok, coerced, _ = _coerce(value, branch)
            if ok:
                return True, coerced, ""
        return False, value, " or ".join(_describe(branch) for branch in branches)

    kind = schema.get("type")
    if kind == "integer":
        ok, coerced = _to_integer(value)
    elif kind == "number":
        ok, coerced = _to_number(value)
    elif kind == "boolean":
        ok, coerced = _to_boolean(value)
    else:
        ok, coerced = True, value

    allowed = _enum(schema)
    if ok and allowed is not None and coerced not in allowed:
        return False, value, f"one of {list(allowed)!r}"
    return ok, coerced, _describe(schema)


def _matches(value: Any, schema: Mapping[str, Any]) -> bool:
    """Whether ``value`` already fits ``schema`` exactly, with no coercion."""
    branches = [branch for branch in schema.get("anyOf") or () if isinstance(branch, Mapping)]
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
            return True, int(text)
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
            return True, int(text)
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


def _parse_float(text: str) -> float | None:
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _describe(schema: Mapping[str, Any]) -> str:
    branches = [branch for branch in schema.get("anyOf") or () if isinstance(branch, Mapping)]
    if branches:
        return " or ".join(_describe(branch) for branch in branches)
    kind = schema.get("type")
    return kind if isinstance(kind, str) else "any value"


def _names(names: Sequence[str]) -> str:
    return ", ".join(repr(name) for name in names)


def _short(value: Any, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
