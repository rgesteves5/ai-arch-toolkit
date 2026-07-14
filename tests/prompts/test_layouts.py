"""Prompt layout, span, and convenience-constructor tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_arch_toolkit.toolkit.prompts import (
    JsonLayout,
    LayoutResult,
    MarkdownLayout,
    Prompt,
    PromptSection,
    SectionSpan,
    SeparatorPolicy,
    TextLayout,
    XmlLayout,
    render_prompt,
)


def sample_prompt() -> Prompt:
    return Prompt(
        sections=(
            PromptSection(name="role", content="Be <helpful> & kind.", order=100),
            PromptSection(
                name="request",
                content='Use "quotes" and Olá.',
                order=200,
                stability="request",
            ),
        )
    )


def test_default_layout_remains_byte_compatible() -> None:
    rendered = render_prompt(sample_prompt())

    assert rendered.text == 'Be <helpful> & kind.\n\nUse "quotes" and Olá.'
    assert rendered.layout == "text"
    assert rendered.stable_prefix == "Be <helpful> & kind."
    assert rendered.section_text("role") == "Be <helpful> & kind."


def test_prompt_from_text_and_render_method() -> None:
    rendered = Prompt.from_text("literal", name="rules").render()

    assert rendered.text == "literal"
    assert rendered.section_names == ("rules",)


def test_prompt_from_sections_and_render_separator_conveniences() -> None:
    prompt = Prompt.from_sections(
        PromptSection(name="a", content="A"),
        PromptSection(name="b", content="B", order=1),
    )
    assert prompt.render(separator="|").text == "A|B"
    assert prompt.render(layout="markdown", separator="|").text == "## a\n\nA|## b\n\nB"
    with pytest.raises(ValueError, match="layout object"):
        prompt.render(layout=TextLayout(), separator="|")


def test_prompt_from_in_memory_resource_supports_selectors_and_custom_serializers() -> None:
    from ai_arch_toolkit.toolkit.resources import Resource, ResourceRef, SerializerRegistry

    class PrefixSerializer:
        name = "prefix"

        def serialize(self, value: object) -> str:
            return f"prefix:{value}"

    raw = b'{"rules": ["short"]}'
    resource = Resource(
        ref=ResourceRef(uri="memory://rules", media_type="application/json"),
        raw=raw,
        media_type="application/json",
        data={"rules": ["short"]},
        text=raw.decode(),
    )
    serializer_registry = SerializerRegistry({"prefix": PrefixSerializer()})
    section = PromptSection.from_resource(
        resource,
        name="rules",
        selector="/rules",
        serialize_as="prefix",
        serializer_registry=serializer_registry,
    )

    assert Prompt.from_resource(resource).render().text == '{"rules": ["short"]}'
    assert Prompt.from_sections(section).render().text == "prefix:['short']"


def test_prompt_and_section_from_file(tmp_path: Path) -> None:
    path = tmp_path / "rules.json"
    path.write_text('{"writing": {"rules": ["short", "clear"]}}')

    section = PromptSection.from_file(
        path,
        name="rules",
        selector="/writing/rules",
        serialize_as="markdown",
        order=100,
    )
    prompt = Prompt.from_file(path, selector="/writing/rules", serialize_as="json")

    assert section.content == "- short\n- clear"
    assert section.metadata["source"] == str(path)
    assert section.metadata["resource_fingerprint"].startswith("sha256:")
    assert prompt.render().text == '[\n  "short",\n  "clear"\n]'


def test_prompt_section_from_binary_requires_text_serialization(tmp_path: Path) -> None:
    path = tmp_path / "content.bin"
    path.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="binary"):
        PromptSection.from_file(path, name="binary")


def test_text_layout_boundary_separator_policy() -> None:
    policy = SeparatorPolicy(
        default="|",
        between={("role", "request"): "\n--- REQUEST ---\n"},
    )

    rendered = render_prompt(sample_prompt(), layout=TextLayout(separator=policy))

    assert rendered.text == ('Be <helpful> & kind.\n--- REQUEST ---\nUse "quotes" and Olá.')


def test_separator_policy_supports_before_after_and_callable() -> None:
    prompt = Prompt.from_sections(
        PromptSection(name="role", content="ROLE"),
        PromptSection(name="request", content="TASK", order=1),
    )
    policy = SeparatorPolicy(
        default="|",
        before={"role": "<"},
        after={"request": ">"},
        resolver=lambda previous, current: f"[{previous.name}->{current.name}]",
    )
    assert prompt.render(layout=TextLayout(separator=policy)).text == "<ROLE[role->request]TASK>"


def test_separator_policy_validation() -> None:
    with pytest.raises(TypeError, match="default"):
        SeparatorPolicy(default=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="boundaries"):
        SeparatorPolicy(between={"bad": "x"})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="separators"):
        SeparatorPolicy(between={("a", "b"): 1})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="between must be a mapping"):
        SeparatorPolicy(between=[])  # type: ignore[arg-type]


def test_markdown_layout_uses_name_or_metadata_title() -> None:
    prompt = Prompt(
        sections=(
            PromptSection(
                name="role",
                content="Architect",
                metadata={"title": "Agent Role"},
            ),
        )
    )

    rendered = prompt.render(layout=MarkdownLayout(heading_level=3))

    assert rendered.text == "### Agent Role\n\nArchitect"
    assert rendered.section_text("role") == rendered.text
    assert rendered.section_spans[0].content_start == len("### Agent Role\n\n")


def test_markdown_layout_without_headings_and_validation() -> None:
    assert (
        Prompt.from_text("content").render(layout=MarkdownLayout(include_headings=False)).text
        == "content"
    )
    with pytest.raises(ValueError, match="between 1 and 6"):
        MarkdownLayout(heading_level=0)
    with pytest.raises(TypeError, match="separator"):
        MarkdownLayout(separator=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="heading_level"):
        MarkdownLayout(heading_level=True)
    with pytest.raises(TypeError, match="include_headings"):
        MarkdownLayout(include_headings=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="title must be a string"):
        Prompt(sections=(PromptSection(name="bad", content="x", metadata={"title": 1}),)).render(
            layout=MarkdownLayout()
        )


def test_xml_layout_escapes_content_and_attributes() -> None:
    rendered = sample_prompt().render(
        layout=XmlLayout(root_tag="instructions", include_stability=True)
    )

    assert rendered.text == (
        '<instructions>\n<section name="role" stability="static">'
        "Be &lt;helpful&gt; &amp; kind.</section>\n"
        '<section name="request" stability="request">Use "quotes" and Olá.</section>\n'
        "</instructions>"
    )
    assert rendered.section_text("role").startswith('<section name="role"')
    role_span = rendered.section_spans[0]
    assert rendered.text[role_span.content_start : role_span.content_end] == (
        "Be &lt;helpful&gt; &amp; kind."
    )
    assert rendered.stable_prefix.endswith("</section>")


def test_xml_layout_can_include_selected_scalar_metadata() -> None:
    prompt = Prompt.from_sections(
        PromptSection(name="rules", content="R", metadata={"audience": "adult", "active": True})
    )
    rendered = prompt.render(layout=XmlLayout(metadata_attributes=("audience", "active")))
    assert '<section name="rules" audience="adult" active="true">R</section>' in rendered.text


def test_xml_all_static_prefix_includes_closing_root() -> None:
    rendered = Prompt.from_text("static").render(layout="xml")
    assert rendered.stable_prefix == rendered.text


def test_xml_empty_prompt_and_tag_validation() -> None:
    assert Prompt().render(layout=XmlLayout()).text == "<prompt></prompt>"
    with pytest.raises(ValueError, match="valid XML tag"):
        XmlLayout(root_tag="not valid")
    with pytest.raises(TypeError, match="separator"):
        XmlLayout(separator=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="include_stability"):
        XmlLayout(include_stability=1)  # type: ignore[arg-type]


def test_json_layout_is_valid_ordered_and_unicode_safe() -> None:
    rendered = sample_prompt().render(layout=JsonLayout(include_stability=True))
    data = json.loads(rendered.text)

    assert data == [
        {"name": "role", "content": "Be <helpful> & kind.", "stability": "static"},
        {
            "name": "request",
            "content": 'Use "quotes" and Olá.',
            "stability": "request",
        },
    ]
    assert rendered.section_text("role").lstrip().startswith('{"name": "role"')


def test_json_compact_and_empty_layouts() -> None:
    assert Prompt().render(layout=JsonLayout()).text == "[]"
    compact = Prompt.from_text("x").render(layout=JsonLayout(indent=None))
    assert compact.text == '[{"name":"prompt","content":"x"}]'
    with pytest.raises(ValueError, match="non-negative"):
        JsonLayout(indent=-1)
    with pytest.raises(ValueError, match="non-negative"):
        JsonLayout(indent=True)
    with pytest.raises(TypeError, match="include_stability"):
        JsonLayout(include_stability=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ensure_ascii"):
        JsonLayout(ensure_ascii=1)  # type: ignore[arg-type]


def test_json_object_layout() -> None:
    prompt = Prompt.from_sections(
        PromptSection(name="role", content="R"),
        PromptSection(name="request", content="Q", order=1, stability="request"),
    )
    rendered = prompt.render(layout=JsonLayout(mode="object", include_stability=True))
    assert json.loads(rendered.text) == {
        "role": {"content": "R", "stability": "static"},
        "request": {"content": "Q", "stability": "request"},
    }
    assert '"role"' in rendered.section_text("role")


@pytest.mark.parametrize("name", ["text", "markdown", "xml", "json"])
def test_builtin_layout_names(name: str) -> None:
    rendered = Prompt.from_text("value").render(layout=name)
    assert rendered.layout == name


def test_unknown_layout_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown prompt layout"):
        Prompt.from_text("value").render(layout="csv")


def test_custom_layout_span_validation() -> None:
    class MissingSpans:
        name = "broken"

        def render(self, sections):
            return LayoutResult(text="x", spans=(), layout=self.name)

    with pytest.raises(ValueError, match="0 spans for 1 sections"):
        Prompt.from_text("x").render(layout=MissingSpans())


def test_custom_layout_span_name_and_bounds_validation() -> None:
    class WrongName:
        name = "broken"

        def render(self, sections):
            return LayoutResult(
                text="x",
                spans=(SectionSpan(name="other", start=0, end=1),),
                layout=self.name,
            )

    class OutOfBounds:
        name = "broken"

        def render(self, sections):
            return LayoutResult(
                text="x",
                spans=(SectionSpan(name="prompt", start=1, end=2),),
                layout=self.name,
            )

    with pytest.raises(ValueError, match="does not match"):
        Prompt.from_text("x").render(layout=WrongName())
    with pytest.raises(ValueError, match="invalid span"):
        Prompt.from_text("x").render(layout=OutOfBounds())


def test_section_span_validation() -> None:
    with pytest.raises(ValueError, match="name is required"):
        SectionSpan(name="", start=0, end=1)
    with pytest.raises(TypeError, match="offsets"):
        SectionSpan(name="x", start=True, end=1)
    with pytest.raises(ValueError, match="invalid section span"):
        SectionSpan(name="x", start=2, end=1)
    with pytest.raises(ValueError, match="both be set"):
        SectionSpan(name="x", start=0, end=1, content_start=0)
    with pytest.raises(ValueError, match="contained"):
        SectionSpan(name="x", start=0, end=1, content_start=0, content_end=2)


def test_layout_result_validation_and_tuple_normalization() -> None:
    span = SectionSpan(name="x", start=0, end=1)
    result = LayoutResult(text="x", spans=[span], layout="custom")  # type: ignore[arg-type]
    assert result.spans == (span,)
    with pytest.raises(TypeError, match="text"):
        LayoutResult(text=1, spans=(), layout="custom")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SectionSpan"):
        LayoutResult(text="x", spans=(object(),), layout="custom")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="layout is required"):
        LayoutResult(text="x", spans=(), layout="")


def test_missing_rendered_section_text_raises() -> None:
    with pytest.raises(KeyError, match="no section"):
        Prompt.from_text("x").render().section_text("missing")
