"""Tests for structured prompt composition and rendering."""

from __future__ import annotations

import hashlib

import pytest

from ai_arch_toolkit import toolkit as toolkit_api
from ai_arch_toolkit.toolkit.knowledge import KnowledgeRegistry
from ai_arch_toolkit.toolkit.prompts import (
    Prompt,
    PromptSection,
    prompt_from_sections,
    render_prompt,
    validate_cache_layout,
)


def test_render_orders_sections_and_preserves_ties() -> None:
    prompt = Prompt(
        sections=(
            PromptSection(name="last", content="LAST", order=200),
            PromptSection(name="first_a", content="FIRST A", order=100),
            PromptSection(name="first_b", content="FIRST B", order=100),
        )
    )

    rendered = render_prompt(prompt)

    assert rendered.section_names == ("first_a", "first_b", "last")
    assert rendered.text == "FIRST A\n\nFIRST B\n\nLAST"
    assert rendered.system == rendered.text


def test_fingerprint_hashes_exact_utf8_text() -> None:
    rendered = render_prompt(Prompt(sections=(PromptSection(name="unicode", content="Olá 🌍"),)))

    expected = hashlib.sha256("Olá 🌍".encode()).hexdigest()
    assert rendered.fingerprint == f"sha256:{expected}"


def test_fingerprint_changes_with_whitespace() -> None:
    first = render_prompt(Prompt(sections=(PromptSection(name="a", content="value"),)))
    second = render_prompt(Prompt(sections=(PromptSection(name="a", content="value "),)))

    assert first.fingerprint != second.fingerprint


def test_metadata_does_not_change_text_fingerprint() -> None:
    first = render_prompt(
        Prompt(sections=(PromptSection(name="rules", content="RULES", metadata={"version": 1}),))
    )
    second = render_prompt(
        Prompt(sections=(PromptSection(name="rules", content="RULES", metadata={"version": 2}),))
    )

    assert first.text == second.text
    assert first.fingerprint == second.fingerprint


def test_empty_prompt_has_empty_text_fingerprint_and_no_stable_prefix() -> None:
    rendered = render_prompt(Prompt())

    expected = hashlib.sha256(b"").hexdigest()
    assert rendered.text == ""
    assert rendered.sections == ()
    assert rendered.section_names == ()
    assert rendered.fingerprint == f"sha256:{expected}"
    assert rendered.stable_prefix_end is None
    assert rendered.stable_prefix == ""


def test_empty_content_is_preserved_as_a_section() -> None:
    rendered = render_prompt(
        Prompt(
            sections=(
                PromptSection(name="empty", content=""),
                PromptSection(name="rules", content="RULES", order=100),
            )
        )
    )

    assert rendered.section_names == ("empty", "rules")
    assert rendered.text == "\n\nRULES"
    assert rendered.stable_prefix == rendered.text


def test_empty_separator_concatenates_sections_without_hidden_whitespace() -> None:
    rendered = render_prompt(
        Prompt(
            sections=(
                PromptSection(name="a", content="A"),
                PromptSection(name="b", content="B", order=100),
            ),
            separator="",
        )
    )

    assert rendered.text == "AB"
    assert rendered.stable_prefix_end == 2
    assert rendered.stable_prefix == "AB"


def test_non_string_separator_is_rejected() -> None:
    with pytest.raises(TypeError, match=r"Prompt\.separator must be a string"):
        Prompt(separator=123)  # type: ignore[arg-type]


def test_duplicate_names_are_rejected() -> None:
    prompt = Prompt(
        sections=(
            PromptSection(name="rules", content="A"),
            PromptSection(name="rules", content="B"),
        )
    )

    with pytest.raises(ValueError, match="duplicates: 'rules'"):
        render_prompt(prompt)


def test_empty_and_non_string_names_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"PromptSection\.name is required"):
        PromptSection(name="", content="value")
    with pytest.raises(ValueError, match=r"PromptSection\.name is required"):
        PromptSection(name=123, content="value")  # type: ignore[arg-type]


def test_render_preserves_order_when_static_content_follows_request_content() -> None:
    prompt = Prompt(
        sections=(
            PromptSection(name="request", content="dynamic", order=100, stability="request"),
            PromptSection(name="rules", content="stable", order=200, stability="static"),
        )
    )

    rendered = render_prompt(prompt)

    assert rendered.section_names == ("request", "rules")
    assert rendered.text == "dynamic\n\nstable"
    assert rendered.stable_prefix_end is None


def test_cache_layout_validation_is_strict_and_opt_in() -> None:
    prompt = Prompt(
        sections=(
            PromptSection(name="request", content="dynamic", order=100, stability="request"),
            PromptSection(name="rules", content="stable", order=200, stability="static"),
        )
    )

    with pytest.raises(ValueError, match=r"static.*follows.*request"):
        validate_cache_layout(prompt)


def test_cache_layout_validation_accepts_static_session_request_order() -> None:
    prompt = Prompt(
        sections=(
            PromptSection(name="rules", content="stable", order=100),
            PromptSection(name="session", content="session", order=200, stability="session"),
            PromptSection(name="request", content="request", order=300, stability="request"),
        )
    )

    assert validate_cache_layout(prompt) is None


def test_stable_prefix_excludes_session_and_request_sections() -> None:
    prompt = Prompt(
        sections=(
            PromptSection(name="role", content="ROLE", order=100),
            PromptSection(name="rules", content="RULES", order=200),
            PromptSection(name="session", content="SESSION", order=300, stability="session"),
            PromptSection(name="request", content="REQUEST", order=400, stability="request"),
        ),
        separator="\n---\n",
    )

    rendered = render_prompt(prompt)

    assert rendered.stable_prefix == "ROLE\n---\nRULES"
    assert rendered.stable_prefix_end == len("ROLE\n---\nRULES")


def test_all_static_sections_make_the_entire_prompt_a_stable_prefix() -> None:
    rendered = render_prompt(
        Prompt(
            sections=(
                PromptSection(name="role", content="ROLE"),
                PromptSection(name="rules", content="RULES", order=100),
            )
        )
    )

    assert rendered.stable_prefix_end == len(rendered.text)
    assert rendered.stable_prefix == rendered.text


def test_no_static_sections_has_no_stable_prefix() -> None:
    rendered = render_prompt(
        Prompt(sections=(PromptSection(name="request", content="dynamic", stability="request"),))
    )

    assert rendered.stable_prefix_end is None
    assert rendered.stable_prefix == ""


def test_position_is_a_compatibility_alias_for_order() -> None:
    section = PromptSection(name="compat", content="value", position=42)

    assert section.order == 42
    assert section.position == 42


def test_order_and_position_cannot_both_be_used() -> None:
    with pytest.raises(ValueError, match="either order or position"):
        PromptSection(name="bad", content="value", order=1, position=2)


def test_invalid_stability_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid prompt stability"):
        PromptSection(name="bad", content="value", stability="daily")  # type: ignore[arg-type]


def test_invalid_content_and_order_types_are_rejected() -> None:
    with pytest.raises(TypeError, match="content must be a string"):
        PromptSection(name="bad", content=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="order must be an integer"):
        PromptSection(name="bad", content="value", order="first")  # type: ignore[arg-type]


def test_metadata_is_copied_and_read_only() -> None:
    original = {
        "source": "test",
        "nested": {"values": ["a"]},
        "tags": {"one", "two"},
    }
    section = PromptSection(name="meta", content="value", metadata=original)
    original["source"] = "changed"
    original["nested"]["values"].append("changed")

    assert section.metadata["source"] == "test"
    assert section.metadata["nested"] == {"values": ("a",)}
    assert section.metadata["tags"] == frozenset({"one", "two"})
    with pytest.raises(TypeError):
        section.metadata["source"] = "blocked"  # type: ignore[index]
    with pytest.raises(TypeError):
        section.metadata["nested"]["values"] = ()  # type: ignore[index]


def test_metadata_rejects_non_string_keys_and_cycles() -> None:
    with pytest.raises(TypeError, match="metadata keys must be strings"):
        PromptSection(name="meta", content="value", metadata={1: "value"})  # type: ignore[dict-item]

    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="metadata cannot contain cycles"):
        PromptSection(name="meta", content="value", metadata=cyclic)

    cyclic_list: list[object] = []
    cyclic_list.append(cyclic_list)
    with pytest.raises(ValueError, match="metadata cannot contain cycles"):
        PromptSection(name="meta", content="value", metadata={"items": cyclic_list})


def test_prompt_sections_and_prompts_are_hashable_without_ignoring_metadata_equality() -> None:
    first = PromptSection(name="rules", content="RULES", metadata={"version": 1})
    second = PromptSection(name="rules", content="RULES", metadata={"version": 2})

    assert first != second
    assert hash(first) == hash(second)
    assert hash(Prompt(sections=(first,))) == hash(Prompt(sections=(second,)))


def test_prompt_from_sections_freezes_a_sequence() -> None:
    sections = [PromptSection(name="one", content="ONE")]
    prompt = prompt_from_sections(sections, separator="\n")
    sections.append(PromptSection(name="two", content="TWO"))

    assert prompt.sections == (PromptSection(name="one", content="ONE"),)
    assert prompt.separator == "\n"


def test_prompt_constructor_freezes_a_mutable_section_sequence() -> None:
    sections = [PromptSection(name="one", content="ONE")]
    prompt = Prompt(sections=sections)  # type: ignore[arg-type]
    sections.append(PromptSection(name="two", content="TWO"))

    assert prompt.sections == (PromptSection(name="one", content="ONE"),)


def test_knowledge_registry_supplies_content_while_prompt_supplies_structure() -> None:
    registry = KnowledgeRegistry()
    registry.register("style", "Be concise.")
    registry.register("domain", "Use architecture terminology.")

    prompt = Prompt(
        sections=(
            PromptSection(name="role", content="You are an architect.", order=100),
            PromptSection(
                name="knowledge",
                content=registry.as_context("style", "domain"),
                order=200,
            ),
        )
    )

    rendered = render_prompt(prompt)

    assert rendered.section_names == ("role", "knowledge")
    assert "Be concise.\n\n---\n\nUse architecture terminology." in rendered.text


def test_toolkit_package_reexports_prompt_api() -> None:
    assert toolkit_api.Prompt is Prompt
    assert toolkit_api.PromptSection is PromptSection
    assert toolkit_api.render_prompt is render_prompt
    assert toolkit_api.validate_cache_layout is validate_cache_layout
