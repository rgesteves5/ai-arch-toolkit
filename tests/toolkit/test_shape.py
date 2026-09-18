"""Declared shapes: each kind checks, names the path, and writes the JSON Schema it checks."""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest

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
    ShapeError,
    Tagged,
    Text,
    Whole,
    json_schema,
    json_type,
)

jsonschema = pytest.importorskip("jsonschema")

# Values of every JSON type, and the Python look-alikes JSON cannot tell apart.
POOL: tuple[object, ...] = (
    None,
    True,
    False,
    0,
    1,
    -1,
    2,
    0.5,
    2.5,
    "",
    "x",
    "keys",
    "text",
    [],
    [""],
    ["x"],
    [1],
    {},
    {"a": 1},
    {"type": "lines"},
    {"type": "lines", "start": 1},
    {"type": "block", "start": 1},
    {"name": "x"},
    {"name": ""},
)


def _message(shape: Shape, value: object, where: str = "field") -> str:
    with pytest.raises(ShapeError) as caught:
        shape.check(value, where)
    return str(caught.value)


@pytest.mark.parametrize(
    ("shape", "good", "bad", "message"),
    [
        (Text(), ["x"], ["", 1, None], "field must be a non-empty string"),
        (Text(empty=True), ["", "x"], [1, None, []], "field must be a string"),
        (Flag(), [True, False], [0, "true", None], "field must be a boolean"),
        (Whole(), [0, -3], [1.0, True, "1", None], "field must be an integer"),
        (Whole(minimum=1), [1, 7], [0, -1, 1.5, True], "field must be a positive integer"),
        (Whole(minimum=0), [0], [-1], "field must be a non-negative integer"),
        (
            Number(above=0),
            [0.5, 3],
            [0, -1, math.inf, math.nan, True, "1"],
            "field must be a finite positive number",
        ),
        (
            Number(minimum=0),
            [0, 2.5],
            [-0.1, math.inf, math.nan],
            "field must be a finite non-negative number",
        ),
        (Number(minimum=0, maximum=2), [0, 2, 1.5], [2.1, -1], "field must be a number between"),
        (
            Choice("keys", "full", "none"),
            ["full"],
            ["all", [], {}, None],
            "field must be one of 'keys', 'full', 'none'",
        ),
        (Choice("none", "strict"), ["strict"], ["weak", 1], "field must be 'none' or 'strict'"),
        (Const(1), [1], [True, 1.0, "1", 2], "field must be 1"),
        (Free(), [{}, {"a": 1}], [[], "x", None], "field must be an object"),
        (Anything(), [None, 1, "x", [], {}], [], ""),
    ],
)
def test_each_kind_takes_its_values_and_names_the_rest(
    shape: Shape, good: list[object], bad: list[object], message: str
) -> None:
    for value in good:
        shape.check(value, "field")
    for value in bad:
        assert message in _message(shape, value)


def test_a_list_names_the_item_that_is_wrong() -> None:
    shape = Items(Text())

    shape.check(["a", "b"])
    assert _message(shape, "a", "include") == "include must be a list"
    assert _message(shape, ["a", ""], "include") == "include[1] must be a non-empty string"


def test_fields_are_closed_named_and_suggested() -> None:
    shape = Fields({"name": Text(), "order": Whole()}, required=frozenset({"name"}))

    shape.check({"name": "x", "order": 2})
    assert _message(shape, {"name": "x", "oder": 1}, "sections[0]") == (
        "sections[0] has unknown fields: 'oder' (did you mean 'order'?)"
    )
    assert _message(shape, {"name": "x", "zzz": 1}, "") == "unknown fields: 'zzz'"
    assert _message(shape, {}, "") == "missing 'name'"
    assert _message(shape, {"order": 1}, "sections[0]") == "sections[0] is missing 'name'"
    assert _message(shape, {"name": "x", "order": "1"}, "s") == "s.order must be an integer"
    assert _message(shape, [], "s") == "s must be an object"


def test_a_required_field_must_be_declared() -> None:
    with pytest.raises(ValueError, match="not declared"):
        Fields({"name": Text()}, required=frozenset({"nmae"}))


def test_names_are_non_empty_strings_and_share_a_shape() -> None:
    shape = Names(Fields({"system": Text()}))

    shape.check({"planner": {"system": "x"}})
    assert _message(shape, {"": {}}, "strategy.phases") == (
        "strategy.phases names must be non-empty strings"
    )
    assert _message(shape, {1: {}}, "phases") == "phases names must be non-empty strings"
    assert _message(shape, {"planner": {"system": 1}}, "phases") == (
        "phases.planner.system must be a non-empty string"
    )


def test_maybe_lets_null_mean_not_set() -> None:
    shape = Maybe(Whole(minimum=1))

    shape.check(None)
    shape.check(3)
    assert _message(shape, 0, "max_iterations") == "max_iterations must be a positive integer"


def test_either_routes_by_json_type_and_keeps_the_inner_message() -> None:
    shape = Either(Text(), Items(Text()))

    shape.check("a.prompt.json")
    shape.check(["a.prompt.json"])
    assert _message(shape, 1, "include") == ("include must be a non-empty string or a list")
    assert _message(shape, [""], "include") == "include[0] must be a non-empty string"
    assert Either(Number(above=0), Text()).check(3) is None  # an int is a number


def test_alternatives_must_not_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        Either(Text(), Choice("a", "b"))
    with pytest.raises(ValueError, match="overlap"):
        Either(Whole(), Number())


def test_a_tagged_object_follows_the_variant_its_tag_names() -> None:
    shape = Tagged(
        "type",
        {
            "lines": Fields({"start": Whole(minimum=0), "end": Maybe(Whole(minimum=0))}),
            "block": Fields({"start_marker": Text()}, required=frozenset({"start_marker"})),
        },
    )

    shape.check({"type": "lines", "start": 1})
    shape.check({"type": "block", "start_marker": "<<"})
    assert _message(shape, {}, "select") == "select is missing 'type'"
    assert _message(shape, {"type": "csv"}, "select") == "select.type must be 'lines' or 'block'"
    assert _message(shape, {"type": "lines", "start": -1}, "select") == (
        "select.start must be a non-negative integer"
    )
    assert "unknown fields: 'start'" in _message(shape, {"type": "block", "start": 1}, "select")
    assert _message(shape, {"type": "block"}, "select") == "select is missing 'start_marker'"


def _section() -> Shape:
    ref = Ref("section")
    return ref.bind(Fields({"name": Text(), "sections": Items(ref)}, required=frozenset({"name"})))


def test_a_reference_lets_a_shape_hold_itself() -> None:
    section = _section()

    section.check({"name": "a", "sections": [{"name": "b", "sections": [{"name": "c"}]}]})
    assert _message(section, {"name": "a", "sections": [{"sections": []}]}, "s") == (
        "s.sections[0] is missing 'name'"
    )


def test_an_unbound_reference_says_so() -> None:
    with pytest.raises(ValueError, match="never bound"):
        Ref("loose").check({})


def test_the_schema_puts_references_under_defs() -> None:
    schema = json_schema(Items(_section()), title="Sections", schema_id="https://x/s.json")

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["type"] == "array"
    assert schema["$defs"] == {
        "section": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "sections": {"type": "array", "items": {"$ref": "#/$defs/section"}},
            },
            "additionalProperties": False,
            "required": ["name"],
        }
    }
    jsonschema.Draft202012Validator.check_schema(schema)


# One of each kind, nested, to prove the schema and the check agree on every value of the pool.
_SAMPLES: dict[str, Shape] = {
    "text": Text(),
    "any_text": Text(empty=True),
    "flag": Flag(),
    "whole": Whole(minimum=1),
    "number": Number(above=0),
    "range": Number(minimum=0, maximum=2),
    "choice": Choice("keys", "full", "none"),
    "const": Const(1),
    "free": Free(),
    "anything": Anything(),
    "maybe": Maybe(Text()),
    "items": Items(Text()),
    "fields": Fields({"name": Text(), "order": Whole()}, required=frozenset({"name"})),
    "names": Names(Items(Whole())),
    "either": Either(Text(), Items(Text()), Fields({"path": Text()})),
    "tagged": Tagged(
        "type",
        {
            "lines": Fields({"start": Whole(minimum=0)}),
            "block": Fields({"start_marker": Text()}),
        },
    ),
    "section": _section(),
}


def _json_can_tell(value: object) -> bool:
    """False for the values JSON has no way to write apart: ``1.0`` and the non-finite floats."""
    if isinstance(value, float):
        return math.isfinite(value) and not value.is_integer()
    if isinstance(value, list):
        return all(_json_can_tell(item) for item in value)
    if isinstance(value, dict):
        return all(_json_can_tell(item) for item in value.values())
    return True


def _pool() -> Iterator[object]:
    yield from (value for value in POOL if _json_can_tell(value))


@pytest.mark.parametrize("name", sorted(_SAMPLES))
def test_the_schema_accepts_exactly_what_the_check_accepts(name: str) -> None:
    shape = _SAMPLES[name]
    validator = jsonschema.Draft202012Validator(
        json_schema(shape, title=name, schema_id=f"https://x/{name}.json")
    )
    for value in _pool():
        try:
            shape.check(value)
        except ShapeError:
            checked = False
        else:
            checked = True
        assert validator.is_valid(value) is checked, (name, value)


def test_json_types_follow_json_not_python() -> None:
    assert [json_type(v) for v in (None, True, 1, 1.5, "", [], (), {})] == [
        "null",
        "boolean",
        "integer",
        "number",
        "string",
        "array",
        "array",
        "object",
    ]
    assert json_type(object()) == "other"
