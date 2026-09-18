"""Documents that probe a declared shape one position at a time: the conformance corpus.

``documents(shape)`` starts from the smallest document the shape accepts and, at every position
the declaration names, tries values of every JSON type and the edges of each rule. It stops at
``depth`` so a shape that holds itself (sections in sections) stays finite.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from ai_arch_toolkit.toolkit._shape import (
    Anything,
    Choice,
    Const,
    Either,
    Fields,
    Flag,
    Free,
    Items,
    Maybe,
    Names,
    Number,
    Ref,
    Shape,
    Tagged,
    Text,
    Whole,
)

# One value of each JSON type, and a few of each that rules tell apart.
POOL: tuple[object, ...] = (None, True, 0, 1, -1, 0.5, "", "x", [], [""], ["x"], {}, {"a": 1})


def minimal(shape: Shape) -> object:
    """The smallest value ``shape`` accepts (for a document: its required fields)."""
    if isinstance(shape, Ref):
        return minimal(shape.resolve())
    if isinstance(shape, Text):
        return "" if shape.empty and shape.pattern is None else "x"
    if isinstance(shape, Whole):
        return shape.minimum if shape.minimum is not None else 0
    if isinstance(shape, Number):
        if shape.minimum is not None:
            return shape.minimum
        return shape.above + 1 if shape.above is not None else 0
    if isinstance(shape, Choice):
        return shape.values[0]
    if isinstance(shape, Const):
        return shape.value
    if isinstance(shape, Fields):
        return {name: minimal(shape.fields[name]) for name in sorted(shape.required)}
    if isinstance(shape, Tagged):
        return minimal(next(iter(shape.variants.values())))
    if isinstance(shape, Either):
        return minimal(shape.options[0])
    if isinstance(shape, Items | Names | Free):
        return [] if isinstance(shape, Items) else {}
    if isinstance(shape, Flag):
        return False
    assert isinstance(shape, Maybe | Anything), shape
    return None


def _edges(shape: Shape) -> tuple[object, ...]:
    """The values next to a rule's limits."""
    if isinstance(shape, Whole):
        limits = [bound for bound in (shape.minimum, shape.maximum) if bound is not None]
        return tuple(value for bound in limits for value in (bound - 1, bound, bound + 1))
    if isinstance(shape, Number):
        limits = [bound for bound in (shape.minimum, shape.maximum, shape.above) if bound]
        return tuple(value for bound in limits for value in (bound - 0.5, bound, bound + 0.5))
    if isinstance(shape, Choice):
        return (*shape.values, "none of these")
    if isinstance(shape, Const):
        return (shape.value, "1")
    if isinstance(shape, Text) and shape.pattern is not None:
        return ("tag", "bad tag")
    return ()


def _variants(shape: Shape, depth: int) -> Iterator[object]:
    """Values for one position: the pool, the rule's edges, and (inside) each child varied."""
    yield from POOL
    yield from _edges(shape)
    if depth <= 0:
        return
    if isinstance(shape, Ref):
        yield from _variants(shape.resolve(), depth - 1)
    elif isinstance(shape, Maybe):
        yield from _variants(shape.shape, depth)
    elif isinstance(shape, Either):
        for option in shape.options:
            yield from _variants(option, depth)
    elif isinstance(shape, Tagged):
        for variant in shape.variants.values():
            yield from _variants(variant, depth)
    elif isinstance(shape, Items):
        yield from ([value] for value in _variants(shape.item, depth - 1))
    elif isinstance(shape, Names):
        yield from ({"p": value} for value in _variants(shape.value, depth - 1))
        yield {"": minimal(shape.value)}
    elif isinstance(shape, Fields):
        base = minimal(shape)
        assert isinstance(base, dict)
        yield {**base, "zz_unknown": 1}
        for name, field in shape.fields.items():
            for value in _variants(field, depth - 1):
                yield {**base, name: value}
            yield {key: value for key, value in base.items() if key != name}


def documents(shape: Shape, *, depth: int = 4) -> list[object]:
    """Distinct documents probing ``shape`` down to ``depth`` levels of nesting."""
    seen: dict[str, object] = {}
    for document in _variants(shape, depth):
        seen.setdefault(json.dumps(document, sort_keys=True), document)
    return list(seen.values())
