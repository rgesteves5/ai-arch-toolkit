"""The declared shape of a configuration document: one declaration checks it and publishes it.

A manifest's shape is declared once, from the kinds below. ``shape.check(document)`` verifies a
loaded document and names the first thing wrong by its path (``strategy.max_iterations``,
``sections[0].source``); ``json_schema(shape, ...)`` writes the same shape as a JSON Schema (draft
2020-12) for editors and other tools. Loaders enforce the declaration, not the schema, so the
schema cannot say anything the declaration does not.

JSON cannot tell ``1`` from ``1.0`` and has no infinity, so the schema is a little looser than the
check in two places: an integer written as ``1.0``, and a non-finite number read from YAML or TOML.
"""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, TypeAliasType, get_args

__all__ = [
    "Anything",
    "Choice",
    "Const",
    "Either",
    "Fields",
    "Flag",
    "Free",
    "Items",
    "Maybe",
    "Names",
    "Number",
    "Ref",
    "Shape",
    "ShapeError",
    "Tagged",
    "Text",
    "Whole",
    "json_schema",
    "json_type",
]

type JsonSchema = dict[str, object] | bool

_ALL_KINDS = frozenset({"array", "boolean", "integer", "null", "number", "object", "string"})


class ShapeError(ValueError):
    """A document that does not have its declared shape. The message starts with the path."""


class Shape(Protocol):
    """One kind of value a document may hold."""

    def check(self, value: object, where: str = "") -> None:
        """Raise ``ShapeError`` unless ``value`` has this shape; ``where`` is its path."""
        ...

    def kinds(self) -> frozenset[str]:
        """The JSON types this shape admits, to route a value among alternatives."""
        ...

    def describe(self) -> str:
        """What this shape is, for an error message ("a non-empty string")."""
        ...

    def schema(self, defs: dict[str, JsonSchema]) -> JsonSchema:
        """This shape as JSON Schema; a named shape inside it adds itself to ``defs`` once."""
        ...


def json_type(value: object) -> str:
    """The JSON type of a loaded value; ``"other"`` for anything JSON has no type for."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list | tuple):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return "other"


def _label(where: str) -> str:
    return where or "the document"


def _fail(where: str, shape: Shape) -> ShapeError:
    return ShapeError(f"{_label(where)} must be {shape.describe()}")


def _missing(where: str, name: str) -> str:
    return f"{where} is missing {name!r}" if where else f"missing {name!r}"


def _at(where: str, name: str) -> str:
    return f"{where}.{name}" if where else name


def _either(words: list[str]) -> str:
    """``a``, ``a or b``, ``a, b or c``."""
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} or {words[-1]}"


@dataclass(frozen=True, slots=True, kw_only=True)
class Text:
    """A string; non-empty unless ``empty``, and matching ``pattern`` when one is given."""

    empty: bool = False
    pattern: str | None = None

    def check(self, value: object, where: str = "") -> None:
        if not isinstance(value, str) or not (self.empty or value):
            raise _fail(where, self)
        if self.pattern is not None and re.fullmatch(self.pattern, value) is None:
            raise _fail(where, self)

    def kinds(self) -> frozenset[str]:
        return frozenset({"string"})

    def describe(self) -> str:
        what = "a string" if self.empty else "a non-empty string"
        return what if self.pattern is None else f"{what} matching {self.pattern!r}"

    def schema(self, defs: dict[str, JsonSchema]) -> JsonSchema:
        schema: dict[str, object] = {"type": "string"}
        if not self.empty:
            schema["minLength"] = 1
        if self.pattern is not None:
            schema["pattern"] = self.pattern
        return schema


@dataclass(frozen=True, slots=True)
class Flag:
    """A boolean."""

    def check(self, value: object, where: str = "") -> None:
        if not isinstance(value, bool):
            raise _fail(where, self)

    def kinds(self) -> frozenset[str]:
        return frozenset({"boolean"})

    def describe(self) -> str:
        return "a boolean"

    def schema(self, defs: dict[str, JsonSchema]) -> JsonSchema:
        return {"type": "boolean"}


@dataclass(frozen=True, slots=True, kw_only=True)
class Whole:
    """An integer (never a boolean or a float), from ``minimum`` to ``maximum``."""

    minimum: int | None = None
    maximum: int | None = None

    def check(self, value: object, where: str = "") -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise _fail(where, self)
        if (self.minimum is not None and value < self.minimum) or (
            self.maximum is not None and value > self.maximum
        ):
            raise _fail(where, self)

    def kinds(self) -> frozenset[str]:
        return frozenset({"integer"})

    def describe(self) -> str:
        if self.minimum is not None and self.maximum is not None:
            return f"an integer between {self.minimum} and {self.maximum}"
        if self.maximum is not None:
            return f"an integer <= {self.maximum}"
        named = {None: "an integer", 0: "a non-negative integer", 1: "a positive integer"}
        return named.get(self.minimum, f"an integer >= {self.minimum}")

    def schema(self, defs: dict[str, JsonSchema]) -> JsonSchema:
        schema: dict[str, object] = {"type": "integer"}
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        return schema


@dataclass(frozen=True, slots=True, kw_only=True)
class Number:
    """A finite number (an int or a float, never a boolean) within the given bounds.

    ``above`` is an exclusive lower bound (``above=0``: positive).
    """

    minimum: float | None = None
    maximum: float | None = None
    above: float | None = None

    def check(self, value: object, where: str = "") -> None:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise _fail(where, self)
        number = float(value)
        if (
            not math.isfinite(number)
            or (self.minimum is not None and number < self.minimum)
            or (self.maximum is not None and number > self.maximum)
            or (self.above is not None and number <= self.above)
        ):
            raise _fail(where, self)

    def kinds(self) -> frozenset[str]:
        return frozenset({"integer", "number"})

    def describe(self) -> str:
        if self.minimum is not None and self.maximum is not None:
            return f"a number between {_plain(self.minimum)} and {_plain(self.maximum)}"
        if (self.above, self.minimum, self.maximum) == (0, None, None):
            return "a finite positive number"
        if (self.above, self.minimum, self.maximum) == (None, 0, None):
            return "a finite non-negative number"
        bounds = ((">", self.above), (">=", self.minimum), ("<=", self.maximum))
        limits = [f"{sign} {_plain(bound)}" for sign, bound in bounds if bound is not None]
        return " ".join(["a finite number", *limits])

    def schema(self, defs: dict[str, JsonSchema]) -> JsonSchema:
        schema: dict[str, object] = {"type": "number"}
        for key, bound in (
            ("minimum", self.minimum),
            ("maximum", self.maximum),
            ("exclusiveMinimum", self.above),
        ):
            if bound is not None:
                schema[key] = bound
        return schema


def _plain(number: float) -> str:
    return str(int(number)) if float(number).is_integer() else str(number)


@dataclass(frozen=True, slots=True, init=False)
class Choice:
    """One of a few strings."""

    values: tuple[str, ...]

    def __init__(self, *values: str) -> None:
        object.__setattr__(self, "values", values)

    @classmethod
    def of(cls, alias: TypeAliasType) -> Choice:
        """The strings of a ``Literal`` alias (``type Mode = Literal["a", "b"]``), in order."""
        values = get_args(alias.__value__)
        if not values or not all(isinstance(value, str) for value in values):
            raise TypeError(f"{alias.__name__} is not a Literal of strings")
        return cls(*(str(value) for value in values))

    def check(self, value: object, where: str = "") -> None:
        if not isinstance(value, str) or value not in self.values:
            raise _fail(where, self)

    def kinds(self) -> frozenset[str]:
        return frozenset({"string"})

    def describe(self) -> str:
        quoted = [repr(value) for value in self.values]
        return _either(quoted) if len(quoted) <= 2 else f"one of {', '.join(quoted)}"

    def schema(self, defs: dict[str, JsonSchema]) -> JsonSchema:
        return {"enum": list(self.values)}


@dataclass(frozen=True, slots=True)
class Const:
    """Exactly this value, of this type (``1``, not ``True`` or ``1.0``)."""

    value: int | str

    def check(self, value: object, where: str = "") -> None:
        if json_type(value) != json_type(self.value) or value != self.value:
            raise _fail(where, self)

    def kinds(self) -> frozenset[str]:
        return frozenset({json_type(self.value)})

    def describe(self) -> str:
        return repr(self.value)

    def schema(self, defs: dict[str, JsonSchema]) -> JsonSchema:
        return {"const": self.value}


@dataclass(frozen=True, slots=True)
class Anything:
    """Any value."""

    def check(self, value: object, where: str = "") -> None:
        """Every value passes."""

    def kinds(self) -> frozenset[str]:
        return _ALL_KINDS

    def describe(self) -> str:
        return "any value"

    def schema(self, defs: dict[str, JsonSchema]) -> JsonSchema:
        return True


@dataclass(frozen=True, slots=True)
class Free:
    """An object whose content is free-form here (metadata, keyword arguments)."""

    def check(self, value: object, where: str = "") -> None:
        if not isinstance(value, Mapping):
            raise _fail(where, self)

    def kinds(self) -> frozenset[str]:
        return frozenset({"object"})

    def describe(self) -> str:
        return "an object"

    def schema(self, defs: dict[str, JsonSchema]) -> JsonSchema:
        return {"type": "object"}


@dataclass(frozen=True, slots=True)
class Maybe:
    """``null``, or a value of ``shape``: an optional field whose ``null`` means "not set"."""

    shape: Shape

    def check(self, value: object, where: str = "") -> None:
        if value is not None:
            self.shape.check(value, where)

    def kinds(self) -> frozenset[str]:
        return self.shape.kinds() | {"null"}

    def describe(self) -> str:
        return self.shape.describe()

    def schema(self, defs: dict[str, JsonSchema]) -> JsonSchema:
        inner = self.shape.schema(defs)
        if not isinstance(inner, dict):
            return inner
        kind, choices, options = inner.get("type"), inner.get("enum"), inner.get("anyOf")
        if isinstance(kind, str):
            return {**inner, "type": [kind, "null"]}
        if isinstance(choices, list):
            return {**inner, "enum": [*choices, None]}
        if isinstance(options, list) and len(inner) == 1:
            return {"anyOf": [*options, {"type": "null"}]}
        return {"anyOf": [inner, {"type": "null"}]}


@dataclass(frozen=True, slots=True)
class Items:
    """A list whose items all have ``item``'s shape."""

    item: Shape

    def check(self, value: object, where: str = "") -> None:
        if not isinstance(value, list | tuple):
            raise _fail(where, self)
        for index, entry in enumerate(value):
            self.item.check(entry, f"{where}[{index}]")

    def kinds(self) -> frozenset[str]:
        return frozenset({"array"})

    def describe(self) -> str:
        return "a list"

    def schema(self, defs: dict[str, JsonSchema]) -> JsonSchema:
        return {"type": "array", "items": self.item.schema(defs)}


@dataclass(frozen=True, slots=True)
class Fields:
    """An object with a closed set of keys, each with its own shape."""

    fields: Mapping[str, Shape]
    required: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        unknown = self.required - set(self.fields)
        if unknown:
            raise ValueError(f"required fields not declared: {sorted(unknown)}")

    def check(self, value: object, where: str = "") -> None:
        if not isinstance(value, Mapping):
            raise _fail(where, self)
        unknown = sorted(str(key) for key in value if key not in self.fields)
        if unknown:
            details = [_suggest(name, self.fields) for name in unknown]
            prefix = f"{where} has unknown fields" if where else "unknown fields"
            raise ShapeError(f"{prefix}: {', '.join(details)}")
        missing = sorted(self.required - set(value))
        if missing:
            raise ShapeError(_missing(where, missing[0]))
        for name, shape in self.fields.items():
            if name in value:
                shape.check(value[name], _at(where, name))

    def kinds(self) -> frozenset[str]:
        return frozenset({"object"})

    def describe(self) -> str:
        return "an object"

    def schema(self, defs: dict[str, JsonSchema]) -> JsonSchema:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {name: shape.schema(defs) for name, shape in self.fields.items()},
            "additionalProperties": False,
        }
        if self.required:
            schema["required"] = sorted(self.required)
        return schema


def _suggest(name: str, known: Iterable[str]) -> str:
    match = difflib.get_close_matches(name, list(known), n=1)
    return f"{name!r}" + (f" (did you mean {match[0]!r}?)" if match else "")


@dataclass(frozen=True, slots=True)
class Names:
    """An object whose keys are names (non-empty strings) and whose values share one shape."""

    value: Shape

    def check(self, value: object, where: str = "") -> None:
        if not isinstance(value, Mapping):
            raise _fail(where, self)
        for name, entry in value.items():
            if not isinstance(name, str) or not name:
                raise ShapeError(f"{_label(where)} names must be non-empty strings")
            self.value.check(entry, _at(where, name))

    def kinds(self) -> frozenset[str]:
        return frozenset({"object"})

    def describe(self) -> str:
        return "an object"

    def schema(self, defs: dict[str, JsonSchema]) -> JsonSchema:
        return {
            "type": "object",
            "propertyNames": {"minLength": 1},
            "additionalProperties": self.value.schema(defs),
        }


@dataclass(frozen=True, slots=True, init=False)
class Either:
    """One of a few shapes, told apart by JSON type: a string path, or a list of them, or ..."""

    options: tuple[Shape, ...]

    def __init__(self, *options: Shape) -> None:
        seen: set[str] = set()
        for option in options:
            overlap = seen & option.kinds()
            if overlap:
                raise ValueError(f"alternatives overlap on {sorted(overlap)}")
            seen |= option.kinds()
        object.__setattr__(self, "options", options)

    def check(self, value: object, where: str = "") -> None:
        kind = json_type(value)
        for option in self.options:
            if kind in option.kinds():
                option.check(value, where)
                return
        raise _fail(where, self)

    def kinds(self) -> frozenset[str]:
        return frozenset().union(*(option.kinds() for option in self.options))

    def describe(self) -> str:
        return _either([option.describe() for option in self.options])

    def schema(self, defs: dict[str, JsonSchema]) -> JsonSchema:
        return {"anyOf": [option.schema(defs) for option in self.options]}


@dataclass(frozen=True, slots=True, init=False)
class Tagged:
    """An object whose ``tag`` key (``type``) says which of ``variants`` its fields follow."""

    tag: str
    variants: Mapping[str, Fields]

    def __init__(self, tag: str, variants: Mapping[str, Fields]) -> None:
        object.__setattr__(self, "tag", tag)
        object.__setattr__(
            self,
            "variants",
            {
                name: Fields(
                    {tag: Const(name), **variant.fields}, required=variant.required | {tag}
                )
                for name, variant in variants.items()
            },
        )

    def check(self, value: object, where: str = "") -> None:
        if not isinstance(value, Mapping):
            raise _fail(where, self)
        if self.tag not in value:
            raise ShapeError(_missing(where, self.tag))
        Choice(*self.variants).check(value[self.tag], _at(where, self.tag))
        self.variants[value[self.tag]].check(value, where)

    def kinds(self) -> frozenset[str]:
        return frozenset({"object"})

    def describe(self) -> str:
        return "an object"

    def schema(self, defs: dict[str, JsonSchema]) -> JsonSchema:
        return {
            "type": "object",
            "required": [self.tag],
            "properties": {self.tag: {"enum": list(self.variants)}},
            "anyOf": [variant.schema(defs) for variant in self.variants.values()],
        }


class Ref:
    """A named shape, written once under ``$defs`` and referenced from each place that uses it.

    It is how a shape holds itself (a section holds sections): create the reference, use it inside
    the shape, then ``bind`` the shape to it.
    """

    __slots__ = ("name", "target")

    def __init__(self, name: str) -> None:
        self.name = name
        self.target: Shape | None = None

    def bind(self, target: Shape) -> Ref:
        """Point this reference at ``target``; returns the reference."""
        self.target = target
        return self

    def resolve(self) -> Shape:
        """The shape this reference stands for."""
        if self.target is None:
            raise ValueError(f"shape reference {self.name!r} was never bound")
        return self.target

    def check(self, value: object, where: str = "") -> None:
        self.resolve().check(value, where)

    def kinds(self) -> frozenset[str]:
        return self.resolve().kinds()

    def describe(self) -> str:
        return self.resolve().describe()

    def schema(self, defs: dict[str, JsonSchema]) -> JsonSchema:
        if self.name not in defs:
            defs[self.name] = True  # held while the target is written: it may refer to itself
            defs[self.name] = self.resolve().schema(defs)
        return {"$ref": f"#/$defs/{self.name}"}


def json_schema(shape: Shape, *, title: str, schema_id: str) -> dict[str, object]:
    """``shape`` as a standalone JSON Schema document, with its named shapes under ``$defs``."""
    document: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": schema_id,
        "title": title,
    }
    defs: dict[str, JsonSchema] = {}
    root = shape.schema(defs)
    if isinstance(root, dict):
        document.update(root)
    if defs:
        document["$defs"] = defs
    return document
