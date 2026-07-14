"""Resource selector and serializer edge cases."""

from __future__ import annotations

import json

import pytest

from ai_arch_toolkit.toolkit.resources import (
    IdentitySelector,
    JsonPointer,
    LineRange,
    MarkdownHeading,
    NamedBlock,
    Resource,
    ResourceSelectorError,
    ResourceSerializationError,
    SerializerRegistry,
    select_resource,
    serialize_resource_value,
)


def structured(data: object) -> Resource:
    return Resource(
        ref=Resource.from_text("{}").ref,
        raw=json.dumps(data).encode(),
        media_type="application/json",
        data=data,
        text=json.dumps(data),
    )


def binary() -> Resource:
    return Resource(
        ref=Resource.from_text("x").ref,
        raw=b"x",
        media_type="application/octet-stream",
        data=b"x",
        text=None,
    )


def test_json_pointer_root_objects_arrays_and_escapes() -> None:
    resource = structured({"a/b": {"m~n": ["zero", "one"]}})

    assert JsonPointer("").select(resource) == resource.data
    assert JsonPointer("/a~1b/m~0n/1").select(resource) == "one"
    assert select_resource(resource, "/a~1b/m~0n/0") == "zero"
    assert IdentitySelector().select(resource) == resource.data


def test_json_pointer_requires_string() -> None:
    with pytest.raises(TypeError, match="pointer must be a string"):
        JsonPointer(1)  # type: ignore[arg-type]


@pytest.mark.parametrize("pointer", ["invalid", "/bad~2escape", "/bad~"])
def test_invalid_json_pointer_syntax(pointer: str) -> None:
    with pytest.raises(ResourceSelectorError):
        JsonPointer(pointer).select(structured({"bad": 1}))


def test_json_pointer_missing_key_reports_available_keys() -> None:
    with pytest.raises(ResourceSelectorError, match="available keys: a, b"):
        JsonPointer("/missing").select(structured({"a": 1, "b": 2}))


@pytest.mark.parametrize("pointer", ["/-", "/01", "/x", "/2"])
def test_json_pointer_invalid_array_index(pointer: str) -> None:
    with pytest.raises(ResourceSelectorError):
        JsonPointer(pointer).select(structured(["a", "b"]))


def test_json_pointer_cannot_traverse_scalar() -> None:
    with pytest.raises(ResourceSelectorError, match="cannot traverse"):
        JsonPointer("/a/b").select(structured({"a": 1}))


def test_string_selector_requires_structured_resource() -> None:
    with pytest.raises(ResourceSelectorError, match="require a structured"):
        select_resource(Resource.from_text("hello"), "/value")


def test_markdown_heading_selects_nested_content() -> None:
    resource = Resource.from_text(
        "# Root\nintro\n## Rules\nfirst\n### Nested\nsecond\n## Next\nlast\n",
        media_type="text/markdown",
    )

    selected = MarkdownHeading(heading="Rules").select(resource)

    assert selected == "first\n### Nested\nsecond"


def test_markdown_heading_can_include_heading() -> None:
    resource = Resource.from_text("# Rules\ncontent\n", media_type="text/markdown")
    assert MarkdownHeading(heading="Rules", include_heading=True).select(resource) == (
        "# Rules\ncontent"
    )


def test_markdown_heading_ambiguity_and_occurrence() -> None:
    resource = Resource.from_text("# Rules\none\n# Rules\ntwo", media_type="text/markdown")
    with pytest.raises(ResourceSelectorError, match="ambiguous"):
        MarkdownHeading(heading="Rules").select(resource)
    assert MarkdownHeading(heading="Rules", occurrence=2).select(resource) == "two"
    with pytest.raises(ResourceSelectorError, match="does not exist"):
        MarkdownHeading(heading="Rules", occurrence=3).select(resource)


def test_markdown_heading_missing_and_validation() -> None:
    resource = Resource.from_text("# Other", media_type="text/markdown")
    with pytest.raises(ResourceSelectorError, match="was not found"):
        MarkdownHeading(heading="Rules").select(resource)
    with pytest.raises(ValueError, match="heading is required"):
        MarkdownHeading(heading="")
    with pytest.raises(ValueError, match="at least 1"):
        MarkdownHeading(heading="Rules", occurrence=0)
    with pytest.raises(TypeError, match="occurrence"):
        MarkdownHeading(heading="Rules", occurrence="first")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="include_heading"):
        MarkdownHeading(heading="Rules", include_heading=1)  # type: ignore[arg-type]
    with pytest.raises(ResourceSelectorError, match="requires text"):
        MarkdownHeading(heading="Rules").select(binary())


def test_line_range_preserves_line_endings() -> None:
    resource = Resource.from_text("one\ntwo\nthree\n")
    assert LineRange(start=2, end=3).select(resource) == "two\nthree\n"
    assert LineRange(start=2).select(resource) == "two\nthree\n"


def test_line_range_validation_and_outside_file() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        LineRange(start=0)
    with pytest.raises(ValueError, match="greater than"):
        LineRange(start=3, end=2)
    with pytest.raises(ResourceSelectorError, match="outside"):
        LineRange(start=2).select(Resource.from_text("one"))
    with pytest.raises(TypeError, match="start must be"):
        LineRange(start=True)
    with pytest.raises(TypeError, match="end must be"):
        LineRange(start=1, end="two")  # type: ignore[arg-type]
    with pytest.raises(ResourceSelectorError, match="requires text"):
        LineRange(start=1).select(binary())


def test_named_block_selection_and_markers() -> None:
    resource = Resource.from_text("before\nBEGIN\ninside\nEND\nafter\n")
    assert NamedBlock(start_marker="BEGIN", end_marker="END").select(resource) == "inside\n"
    assert (
        NamedBlock(start_marker="BEGIN", end_marker="END", include_markers=True).select(resource)
        == "BEGIN\ninside\nEND\n"
    )


def test_named_block_requires_unique_ordered_markers() -> None:
    with pytest.raises(ResourceSelectorError, match="start marker"):
        NamedBlock(start_marker="BEGIN", end_marker="END").select(Resource.from_text("none"))
    with pytest.raises(ResourceSelectorError, match="end marker"):
        NamedBlock(start_marker="BEGIN", end_marker="END").select(
            Resource.from_text("BEGIN\nnone")
        )
    with pytest.raises(ValueError, match="start_marker"):
        NamedBlock(start_marker="", end_marker="END")
    with pytest.raises(ValueError, match="end_marker"):
        NamedBlock(start_marker="BEGIN", end_marker="")
    with pytest.raises(ValueError, match="different"):
        NamedBlock(start_marker="MARK", end_marker="MARK")
    with pytest.raises(TypeError, match="include_markers"):
        NamedBlock(start_marker="BEGIN", end_marker="END", include_markers=1)  # type: ignore[arg-type]
    with pytest.raises(ResourceSelectorError, match="requires text"):
        NamedBlock(start_marker="BEGIN", end_marker="END").select(binary())


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        (True, "true"),
        (False, "false"),
        (1.5, "1.5"),
        ({"name": "Ada", "age": 36}, "name: Ada\nage: 36"),
        ({"empty": None}, "empty: "),
        (["one", "two"], "one\ntwo"),
        (("one", {"two": 2}), 'one\n{"two": 2}'),
        (object(), None),
    ],
)
def test_text_serializer(value: object, expected: str | None) -> None:
    rendered = serialize_resource_value(value, as_format="text")
    assert rendered == expected if expected is not None else rendered.startswith("<object object")


def test_json_serializer_is_pretty_and_unicode_safe() -> None:
    text = serialize_resource_value({"greeting": "Olá"}, as_format="json")
    assert text == '{\n  "greeting": "Olá"\n}'


def test_json_serializer_wraps_unsupported_values() -> None:
    with pytest.raises(ResourceSerializationError, match="not JSON serializable"):
        serialize_resource_value({"value": object()}, as_format="json")


def test_markdown_serializer_handles_nested_values() -> None:
    text = serialize_resource_value(
        {"rules": ["Be concise", {"citations": True}]}, as_format="markdown"
    )
    assert "- **rules**:" in text
    assert "  - Be concise" in text
    assert "    - **citations**: true" in text
    assert serialize_resource_value("plain", as_format="markdown") == "plain"


def test_yaml_serializer_and_dependency_error() -> None:
    assert "greeting: Olá" in serialize_resource_value({"greeting": "Olá"}, as_format="yaml")


def test_yaml_serializer_dependency_and_serialization_errors(monkeypatch) -> None:
    from unittest.mock import patch

    with patch.dict("sys.modules", {"yaml": None}), pytest.raises(ImportError, match=r"\[yaml\]"):
        serialize_resource_value({"x": 1}, as_format="yaml")

    import yaml

    def fail(*args, **kwargs):
        raise yaml.YAMLError("broken")

    monkeypatch.setattr(yaml, "safe_dump", fail)
    with pytest.raises(ResourceSerializationError, match="cannot be serialized as YAML"):
        serialize_resource_value({"x": 1}, as_format="yaml")


def test_unknown_serializer_lists_choices() -> None:
    with pytest.raises(ResourceSerializationError, match="expected one of"):
        serialize_resource_value("value", as_format="xml")


def test_custom_serializer_object_registry_and_return_validation() -> None:
    class PrefixSerializer:
        name = "prefix"

        def serialize(self, value):
            return f"value={value}"

    class InvalidSerializer:
        name = "invalid"

        def serialize(self, value):
            return 1

    registry = SerializerRegistry({"prefix": PrefixSerializer()})
    assert serialize_resource_value("x", as_format="prefix", registry=registry) == "value=x"
    assert serialize_resource_value("x", as_format=PrefixSerializer()) == "value=x"
    assert registry.names == ("json", "markdown", "prefix", "text", "yaml")
    with pytest.raises(ResourceSerializationError, match="must return a string"):
        serialize_resource_value("x", as_format=InvalidSerializer())


def test_binary_cannot_be_serialized_as_text() -> None:
    with pytest.raises(ResourceSerializationError, match="binary"):
        serialize_resource_value(b"bytes", as_format="text")
