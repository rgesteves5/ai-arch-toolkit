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
  ``"1"`` as a string). A branch is read with its parent's other keywords, so the branches of
  ``{"type": "integer", "oneOf": [{"minimum": 1}, ...]}`` take integers; a value with more than
  1,000 alternatives once the branches are flattened passes as it came, like an untyped one;
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
false`` refuses them, D62). Finally the arguments must bind to the function's signature. A
refusal names the argument, what was expected and what came, in at most 1,000 characters.
"""

from __future__ import annotations

import inspect
import itertools
import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, get_type_hints

from ai_arch_toolkit.core._tools._schema import _hint_to_json_schema

_INTEGER_TEXT = re.compile(r"^[+-]?\d+$")
_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
_COMBINATORS = ("anyOf", "oneOf")

# The alternatives one value is checked against, its branches flattened. Branches multiply (a
# ``type`` list on a parent of n ``anyOf`` branches makes one alternative per pair), and a schema
# that arrives from an MCP server may be hostile, so the walk stops here: a value with more
# alternatives passes as it came, as for an untyped schema, for the tool (or the server behind
# it) to check. A titled enum of a few hundred values (MCP's ``oneOf`` of ``const`` branches:
# countries, currencies, time zones) is still walked whole, and refusing a malformed call takes
# a couple of milliseconds, however the alternatives multiply (measured 2026-10-08, 1,000
# branches sharing a 20,000-value enum included).
_ALTERNATIVES_LIMIT = 1_000

# A refusal goes back to the model, which reads it to correct its call. What was expected shows up
# to 300 characters (a couple of dozen enum values: the schema the model holds lists them all), so
# that what came always shows too; the whole message stops at 1,000 characters (about 250
# tokens), however large the schema or the call.
_EXPECTED_LIMIT = 300
_MESSAGE_LIMIT = 1_000


class ArgumentError(Exception):
    """Tool-call arguments that do not fit the tool's schema or signature.

    The message is cut at 1,000 characters: it goes back to the model.
    """

    def __init__(self, message: str, argument: str | None = None) -> None:
        super().__init__(_cut(message, _MESSAGE_LIMIT))
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
        if not isinstance(declared, Mapping):
            continue
        if value is None and not _admits_none(known, name, declared):
            raise ArgumentError(
                f"argument {name!r}: expected {_describe(declared)}, got null", name
            )
        coerced[name] = _checked(name, value, declared)

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


def _checked(name: str, value: Any, declared: Mapping[str, Any]) -> Any:
    """``value`` coerced to its declared schema and inside its bounds, else ``ArgumentError``."""
    ok, coerced = _coerce(value, declared)
    if ok and _within(coerced, declared):
        return coerced
    expected = _describe_bounds(declared) if ok else _expected(value, declared)
    got = f"{type(value).__name__} {_short(value)}"
    raise ArgumentError(f"argument {name!r}: expected {expected}, got {got}", name)


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
    alternatives = _alternatives(schema) or [schema]
    kinds = [a.get("type") for a in alternatives if a.get("type") in ("integer", "number")]
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
    """Whether a schema lets ``null`` through: an alternative of it that is untyped or typed
    ``null`` (with ``None`` in its ``enum``, if it has one), or more alternatives than the walk
    takes."""
    alternatives = _alternatives(schema)
    if alternatives is None:
        return True
    return any(_takes_null(alternative) for alternative in alternatives)


def _takes_null(schema: Mapping[str, Any]) -> bool:
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


def _branches(schema: Mapping[str, Any]) -> Iterator[Mapping[str, Any]] | None:
    """The alternatives a schema offers one level down, or ``None`` when it offers none: its
    ``anyOf`` branches, else its ``oneOf`` ones, else one per member of a ``type`` list (D62).

    A branch is read with its parent's other keywords, its own winning: ``{"type": "integer",
    "oneOf": [{"minimum": 1}, {"const": 0}]}`` offers integers. A ``oneOf`` is read as an
    ``anyOf`` (a value that fits two branches is the server's to refuse), and a schema with both
    is read by its ``anyOf``.
    """
    for keyword in _COMBINATORS:
        listed = schema.get(keyword)
        if isinstance(listed, list | tuple) and any(isinstance(b, Mapping) for b in listed):
            rest = {key: value for key, value in schema.items() if key not in _COMBINATORS}
            return ({**rest, **branch} for branch in listed if isinstance(branch, Mapping))
    kinds = schema.get("type")
    if isinstance(kinds, list) and kinds:
        rest = {key: value for key, value in schema.items() if key != "type"}
        return ({**rest, "type": kind} for kind in kinds)
    return None


def _alternatives(schema: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    """The schemas without branches that a value of ``schema`` may fit, in order and each kind
    once, or ``None`` when the branches flatten to more than ``_ALTERNATIVES_LIMIT``. A schema
    without branches is its own."""
    found = list(itertools.islice(_leaves(schema), _ALTERNATIVES_LIMIT + 1))
    if len(found) > _ALTERNATIVES_LIMIT:
        return None
    kinds: dict[tuple[type, str | None, int], Mapping[str, Any]] = {}
    for leaf in found:
        kinds.setdefault(_decisive(leaf), leaf)
    return list(kinds.values())


def _decisive(schema: Mapping[str, Any]) -> tuple[type, str | None, int]:
    """What decides whether a value fits a schema without branches: its type, and its ``enum`` by
    identity, since the branches of one parent share the parent's (a long enum is read once)."""
    kind = schema.get("type")
    return type(kind), kind if isinstance(kind, str) else None, id(_enum(schema))


def _leaves(schema: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    branches = _branches(schema)
    if branches is None:
        yield schema
        return
    for branch in branches:
        yield from _leaves(branch)


def _coerce(value: Any, schema: Mapping[str, Any]) -> tuple[bool, Any]:
    """``(ok, coerced value)``. A value that already fits an alternative of the schema as it is
    stays as it is; otherwise the first alternative that converts it decides."""
    if value is None:
        return True, None
    if _branches(schema) is None:
        return _convert(value, schema)
    alternatives = _alternatives(schema)
    if alternatives is None:
        return True, value  # more than the walk takes: as it came, like an untyped schema
    if any(_matches(value, alternative) for alternative in alternatives):
        return True, value
    for alternative in alternatives:
        ok, coerced = _convert(value, alternative)
        if ok:
            return True, coerced
    return False, value


def _convert(value: Any, schema: Mapping[str, Any]) -> tuple[bool, Any]:
    """``value`` converted to the type of a schema without branches, and inside its ``enum``."""
    ok, coerced = _converter(schema)(value)
    allowed = _enum(schema)
    if ok and allowed is not None and coerced not in allowed:
        return False, value
    return ok, coerced


def _converter(schema: Mapping[str, Any]) -> Callable[[Any], tuple[bool, Any]]:
    kind = schema.get("type")
    return _CONVERTERS.get(kind, _as_it_is) if isinstance(kind, str) else _as_it_is


def _matches(value: Any, schema: Mapping[str, Any]) -> bool:
    """Whether ``value`` already fits a schema without branches exactly, with no coercion."""
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


def _expected(value: Any, schema: Mapping[str, Any]) -> str:
    """What ``schema`` takes, for the refusal of ``value``: the values its ``enum`` allows when
    only the enum refused it, else its types."""
    allowed = _enum(schema)
    if allowed is not None and _branches(schema) is None and _converter(schema)(value)[0]:
        shown = list(allowed[:_EXPECTED_LIMIT])  # each takes a character at least
        return _cut(f"one of {shown!r}", _EXPECTED_LIMIT)
    return _describe(schema)


def _describe(schema: Mapping[str, Any]) -> str:
    """The types ``schema`` takes, each named once: ``integer or null``."""
    kinds = dict.fromkeys(_kind(alternative) for alternative in _alternatives(schema) or [schema])
    return _cut(" or ".join(kinds), _EXPECTED_LIMIT)


def _kind(schema: Mapping[str, Any]) -> str:
    kind = schema.get("type")
    return kind if isinstance(kind, str) else "any value"


def _names(names: Sequence[str]) -> str:
    return ", ".join(repr(name) for name in names)


def _short(value: Any, limit: int = 60) -> str:
    return _cut(repr(value), limit)


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."
