"""Tests for nested prompt sections across rendering, templates, and manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ai_arch_toolkit._cli import main
from ai_arch_toolkit.toolkit.prompts import (
    JsonLayout,
    LiteralSource,
    MarkdownLayout,
    Prompt,
    PromptSection,
    PromptTemplate,
    PromptTemplateSection,
    PromptValidationError,
    PromptVariable,
    load_prompt,
    render_prompt,
    validate_cache_layout,
)


def _tree() -> Prompt:
    return Prompt(
        sections=(
            PromptSection(name="task", content="TASK", order=200, stability="request"),
            PromptSection(
                name="context",
                content="CONTEXT",
                order=100,
                sections=(
                    PromptSection(name="examples", content="EXAMPLES", order=200),
                    PromptSection(
                        name="rules",
                        content="RULES",
                        order=100,
                        sections=(PromptSection(name="tone", content="TONE"),),
                    ),
                ),
            ),
        )
    )


def test_walk_orders_each_sibling_level_and_reports_preorder_names() -> None:
    rendered = render_prompt(_tree())

    assert rendered.section_names == ("context", "rules", "tone", "examples", "task")
    assert tuple(section.name for section in rendered.sections) == ("context", "task")


def test_text_layout_flattens_tree_in_preorder() -> None:
    rendered = render_prompt(_tree())

    assert rendered.text == "CONTEXT\n\nRULES\n\nTONE\n\nEXAMPLES\n\nTASK"


def test_markdown_layout_deepens_headings_per_level() -> None:
    rendered = render_prompt(_tree(), layout="markdown")

    assert rendered.text == (
        "## context\n\nCONTEXT\n\n"
        "### rules\n\nRULES\n\n"
        "#### tone\n\nTONE\n\n"
        "### examples\n\nEXAMPLES\n\n"
        "## task\n\nTASK"
    )


def test_markdown_layout_rejects_headings_beyond_level_six() -> None:
    prompt = Prompt(
        sections=(
            PromptSection(
                name="deep",
                content="D",
                sections=(PromptSection(name="deeper", content="E"),),
            ),
        )
    )

    with pytest.raises(ValueError, match="exceeds 6"):
        render_prompt(prompt, layout=MarkdownLayout(heading_level=6))

    without_headings = render_prompt(
        prompt, layout=MarkdownLayout(heading_level=6, include_headings=False)
    )
    assert without_headings.text == "D\n\nE"


def test_xml_layout_nests_child_elements() -> None:
    rendered = render_prompt(_tree(), layout="xml")

    assert rendered.text == (
        "<prompt>\n"
        '<section name="context">CONTEXT\n'
        '<section name="rules">RULES\n'
        '<section name="tone">TONE</section>\n'
        "</section>\n"
        '<section name="examples">EXAMPLES</section>\n'
        "</section>\n"
        '<section name="task">TASK</section>\n'
        "</prompt>"
    )


def test_json_array_layout_nests_sections_key_only_when_present() -> None:
    rendered = render_prompt(_tree(), layout="json")

    assert rendered.text == (
        "[\n"
        '  {"name": "context", "content": "CONTEXT", "sections": '
        '[{"name": "rules", "content": "RULES", "sections": '
        '[{"name": "tone", "content": "TONE"}]}, '
        '{"name": "examples", "content": "EXAMPLES"}]},\n'
        '  {"name": "task", "content": "TASK"}\n'
        "]"
    )
    assert json.loads(rendered.text)[0]["sections"][0]["sections"][0]["name"] == "tone"


def test_json_compact_array_layout_matches_json_dumps() -> None:
    rendered = render_prompt(_tree(), layout=JsonLayout(indent=None))

    expected = json.dumps(
        [
            {
                "name": "context",
                "content": "CONTEXT",
                "sections": [
                    {
                        "name": "rules",
                        "content": "RULES",
                        "sections": [{"name": "tone", "content": "TONE"}],
                    },
                    {"name": "examples", "content": "EXAMPLES"},
                ],
            },
            {"name": "task", "content": "TASK"},
        ],
        separators=(",", ":"),
    )
    assert rendered.text == expected


def test_json_object_layout_nests_named_values() -> None:
    rendered = render_prompt(_tree(), layout=JsonLayout(mode="object"))

    assert json.loads(rendered.text) == {
        "context": {
            "content": "CONTEXT",
            "sections": {
                "rules": {"content": "RULES", "sections": {"tone": "TONE"}},
                "examples": "EXAMPLES",
            },
        },
        "task": "TASK",
    }
    assert rendered.section_names == ("context", "rules", "tone", "examples", "task")


def test_spans_carry_depth_and_parents_contain_children() -> None:
    rendered = render_prompt(_tree(), layout="markdown")
    spans = {span.name: span for span in rendered.section_spans}

    assert [(span.name, span.depth) for span in rendered.section_spans] == [
        ("context", 0),
        ("rules", 1),
        ("tone", 2),
        ("examples", 1),
        ("task", 0),
    ]
    context = spans["context"]
    for child_name in ("rules", "tone", "examples"):
        child = spans[child_name]
        assert context.start <= child.start and child.end <= context.end
    assert context.content_end is not None
    assert spans["rules"].start >= context.content_end
    assert spans["task"].start >= context.end


def test_section_text_returns_whole_subtree_for_parents() -> None:
    rendered = render_prompt(_tree(), layout="markdown")

    assert rendered.section_text("context") == (
        "## context\n\nCONTEXT\n\n"
        "### rules\n\nRULES\n\n"
        "#### tone\n\nTONE\n\n"
        "### examples\n\nEXAMPLES"
    )
    assert rendered.section_text("rules") == "### rules\n\nRULES\n\n#### tone\n\nTONE"
    assert rendered.section_text("tone") == "#### tone\n\nTONE"


def test_stable_prefix_ends_at_parent_content_when_first_child_is_dynamic() -> None:
    prompt = Prompt(
        sections=(
            PromptSection(
                name="parent",
                content="P",
                sections=(PromptSection(name="child", content="C", stability="request"),),
            ),
        )
    )

    rendered = render_prompt(prompt, layout="markdown")

    assert rendered.stable_prefix == "## parent\n\nP"


def test_stable_prefix_covers_fully_static_subtree_before_dynamic_sibling() -> None:
    prompt = Prompt(
        sections=(
            PromptSection(
                name="parent",
                content="P",
                order=100,
                sections=(PromptSection(name="child", content="C"),),
            ),
            PromptSection(name="request", content="R", order=200, stability="request"),
        )
    )

    markdown = render_prompt(prompt, layout="markdown")
    assert markdown.stable_prefix == "## parent\n\nP\n\n### child\n\nC"

    xml = render_prompt(prompt, layout="xml")
    assert xml.stable_prefix == (
        '<prompt>\n<section name="parent">P\n<section name="child">C</section>\n</section>'
    )


def test_stable_prefix_tree_edge_cases() -> None:
    all_static = render_prompt(
        Prompt(
            sections=(
                PromptSection(
                    name="a", content="A", sections=(PromptSection(name="b", content="B"),)
                ),
            )
        )
    )
    assert all_static.stable_prefix_end == len(all_static.text)

    dynamic_root = render_prompt(
        Prompt(
            sections=(
                PromptSection(
                    name="a",
                    content="A",
                    stability="session",
                    sections=(PromptSection(name="b", content="B"),),
                ),
            )
        )
    )
    assert dynamic_root.stable_prefix_end is None


def test_validate_cache_layout_walks_the_tree() -> None:
    validate_cache_layout(
        Prompt(
            sections=(
                PromptSection(
                    name="static",
                    content="S",
                    sections=(PromptSection(name="child", content="C", stability="session"),),
                ),
                PromptSection(name="request", content="R", stability="request"),
            )
        )
    )

    with pytest.raises(ValueError, match="must progress"):
        validate_cache_layout(
            Prompt(
                sections=(
                    PromptSection(
                        name="parent",
                        content="P",
                        sections=(PromptSection(name="child", content="C", stability="request"),),
                    ),
                    PromptSection(name="late_static", content="L"),
                )
            )
        )


def test_duplicate_names_rejected_across_nesting_levels() -> None:
    prompt = Prompt(
        sections=(
            PromptSection(
                name="dup",
                content="A",
                sections=(
                    PromptSection(
                        name="inner",
                        content="B",
                        sections=(PromptSection(name="dup", content="C"),),
                    ),
                ),
            ),
        )
    )

    with pytest.raises(ValueError, match="duplicates: 'dup'"):
        render_prompt(prompt)


def test_prompt_section_children_must_be_sections() -> None:
    with pytest.raises(TypeError, match="sections\\[0\\] must be a PromptSection"):
        PromptSection(name="parent", content="P", sections=("child",))  # type: ignore[arg-type]


def test_template_compile_builds_nested_prompt_sections() -> None:
    template = PromptTemplate(
        sections=(
            PromptTemplateSection.literal(
                name="context",
                content="CONTEXT",
                sections=(PromptTemplateSection.literal(name="rules", content="RULES"),),
            ),
        )
    )

    prompt = template.compile()

    assert prompt.sections[0].sections[0].name == "rules"
    rendered = prompt.render(layout="markdown")
    assert rendered.text == "## context\n\nCONTEXT\n\n### rules\n\nRULES"


def test_template_validate_rejects_duplicates_across_levels() -> None:
    template = PromptTemplate(
        sections=(
            PromptTemplateSection.literal(
                name="dup",
                content="A",
                sections=(PromptTemplateSection.literal(name="dup", content="B"),),
            ),
        )
    )

    with pytest.raises(ValueError, match="duplicates: 'dup'"):
        template.validate()


def test_flat_template_definition_fingerprint_matches_legacy_payload() -> None:
    template = PromptTemplate(
        sections=(
            PromptTemplateSection.literal(name="role", content="ROLE", order=100),
            PromptTemplateSection.literal(
                name="rules", content="RULES", order=200, stability="session"
            ),
        ),
        variables=(PromptVariable(name="x", value_type="string", required=True),),
        name="legacy",
        description="d",
    )

    def legacy_section(name: str, content: str, order: int, stability: str) -> dict[str, object]:
        return {
            "name": name,
            "order": order,
            "stability": stability,
            "engine": "NoneType",
            "provenance": dict(LiteralSource(content).describe()),
            "metadata": {},
        }

    payload = {
        "name": "legacy",
        "description": "d",
        "separator": "\n\n",
        "layout": repr(None),
        "allow_extra_variables": False,
        "metadata": {},
        "sections": [
            legacy_section("role", "ROLE", 100, "static"),
            legacy_section("rules", "RULES", 200, "session"),
        ],
        "variables": [
            {
                "name": "x",
                "type": "string",
                "required": True,
                "has_default": False,
                "default": None,
                "description": "",
                "json_schema": None,
            }
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()

    assert template.fingerprint == "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_nested_template_changes_definition_fingerprint_and_inspect() -> None:
    flat = PromptTemplate(sections=(PromptTemplateSection.literal(name="context", content="C"),))
    nested = PromptTemplate(
        sections=(
            PromptTemplateSection.literal(
                name="context",
                content="C",
                sections=(PromptTemplateSection.literal(name="rules", content="R"),),
            ),
        )
    )

    assert flat.fingerprint != nested.fingerprint
    assert "sections" not in flat.inspect()["sections"][0]
    assert nested.inspect()["sections"][0]["sections"][0]["name"] == "rules"
    assert [source["kind"] for source in nested.sources] == ["literal", "literal"]


def _nested_manifest_yaml() -> str:
    return (
        "version: 1\n"
        "name: nested\n"
        "layout: markdown\n"
        "sections:\n"
        "  - name: context\n"
        "    content: CONTEXT\n"
        "    order: 100\n"
        "    sections:\n"
        "      - name: rules\n"
        "        content: RULES\n"
        "  - name: task\n"
        "    content: TASK\n"
        "    order: 200\n"
    )


def _nested_manifest_data() -> dict[str, object]:
    return {
        "version": 1,
        "name": "nested",
        "layout": "markdown",
        "sections": [
            {
                "name": "context",
                "content": "CONTEXT",
                "order": 100,
                "sections": [{"name": "rules", "content": "RULES"}],
            },
            {"name": "task", "content": "TASK", "order": 200},
        ],
    }


def _nested_manifest_toml() -> str:
    return (
        "version = 1\n"
        'name = "nested"\n'
        'layout = "markdown"\n'
        "\n"
        "[[sections]]\n"
        'name = "context"\n'
        'content = "CONTEXT"\n'
        "order = 100\n"
        "\n"
        "[[sections.sections]]\n"
        'name = "rules"\n'
        'content = "RULES"\n'
        "\n"
        "[[sections]]\n"
        'name = "task"\n'
        'content = "TASK"\n'
        "order = 200\n"
    )


def test_nested_manifests_render_identically_across_formats(tmp_path: Path) -> None:
    (tmp_path / "a.prompt.yaml").write_text(_nested_manifest_yaml())
    (tmp_path / "b.prompt.json").write_text(json.dumps(_nested_manifest_data()))
    (tmp_path / "c.prompt.toml").write_text(_nested_manifest_toml())

    rendered = [
        load_prompt(tmp_path / name).render()
        for name in ("a.prompt.yaml", "b.prompt.json", "c.prompt.toml")
    ]

    expected = "## context\n\nCONTEXT\n\n### rules\n\nRULES\n\n## task\n\nTASK"
    assert [item.text for item in rendered] == [expected] * 3
    assert rendered[0].section_names == ("context", "rules", "task")


def test_manifest_pure_container_section(tmp_path: Path) -> None:
    (tmp_path / "container.prompt.json").write_text(
        json.dumps(
            {
                "version": 1,
                "layout": "markdown",
                "sections": [
                    {
                        "name": "wrapper",
                        "sections": [{"name": "a", "content": "A"}],
                    }
                ],
            }
        )
    )

    rendered = load_prompt(tmp_path / "container.prompt.json").render()

    assert rendered.section_names == ("wrapper", "a")
    assert rendered.text.startswith("## wrapper")
    assert "### a\n\nA" in rendered.text


def test_manifest_merge_applies_nested_operations(tmp_path: Path) -> None:
    (tmp_path / "base.prompt.json").write_text(json.dumps(_nested_manifest_data()))
    (tmp_path / "child.prompt.json").write_text(
        json.dumps(
            {
                "version": 1,
                "extends": "base.prompt.json",
                "sections": [
                    {
                        "name": "context",
                        "merge": True,
                        "sections": [
                            {"name": "examples", "content": "EXAMPLES"},
                            {"name": "rules", "replace": True, "content": "NEW RULES"},
                        ],
                    }
                ],
            }
        )
    )

    rendered = load_prompt(tmp_path / "child.prompt.json").render()

    assert rendered.section_names == ("context", "rules", "examples", "task")
    assert "### rules\n\nNEW RULES" in rendered.text
    assert "### examples\n\nEXAMPLES" in rendered.text


def test_manifest_replace_and_remove_reach_nested_sections(tmp_path: Path) -> None:
    (tmp_path / "base.prompt.json").write_text(json.dumps(_nested_manifest_data()))
    (tmp_path / "child.prompt.json").write_text(
        json.dumps(
            {
                "version": 1,
                "extends": "base.prompt.json",
                "sections": [
                    {"name": "rules", "replace": True, "content": "REPLACED"},
                ],
            }
        )
    )
    (tmp_path / "pruned.prompt.json").write_text(
        json.dumps(
            {
                "version": 1,
                "extends": "base.prompt.json",
                "sections": [{"name": "rules", "remove": True}],
            }
        )
    )

    replaced = load_prompt(tmp_path / "child.prompt.json").render()
    assert "### rules\n\nREPLACED" in replaced.text

    pruned = load_prompt(tmp_path / "pruned.prompt.json").render()
    assert pruned.section_names == ("context", "task")


def test_manifest_nested_validation_errors(tmp_path: Path) -> None:
    def load(sections: list[dict[str, object]], extends: str | None = None) -> None:
        payload: dict[str, object] = {"version": 1, "sections": sections}
        if extends is not None:
            payload["extends"] = extends
        path = tmp_path / "invalid.prompt.json"
        path.write_text(json.dumps(payload))
        load_prompt(path)

    (tmp_path / "base.prompt.json").write_text(json.dumps(_nested_manifest_data()))

    with pytest.raises(PromptValidationError, match="cannot merge into unknown"):
        load(
            [{"name": "ghost", "merge": True, "sections": [{"name": "a", "content": "A"}]}],
            extends="base.prompt.json",
        )
    with pytest.raises(PromptValidationError, match="may only define sections"):
        load(
            [
                {
                    "name": "context",
                    "merge": True,
                    "content": "X",
                    "sections": [{"name": "a", "content": "A"}],
                }
            ],
            extends="base.prompt.json",
        )
    with pytest.raises(PromptValidationError, match="non-empty sections list"):
        load([{"name": "context", "merge": True}], extends="base.prompt.json")
    with pytest.raises(PromptValidationError, match="exactly one of"):
        load([{"name": "empty"}])
    with pytest.raises(PromptValidationError, match="cannot define sections"):
        load(
            [
                {
                    "name": "context",
                    "remove": True,
                    "sections": [{"name": "a", "content": "A"}],
                }
            ],
            extends="base.prompt.json",
        )
    with pytest.raises(PromptValidationError, match="cannot use"):
        load(
            [
                {
                    "name": "top",
                    "content": "T",
                    "sections": [{"name": "nested", "replace": True, "content": "X"}],
                }
            ]
        )


def test_included_nested_duplicates_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "base.prompt.json").write_text(json.dumps(_nested_manifest_data()))
    (tmp_path / "extra.prompt.json").write_text(
        json.dumps(
            {
                "version": 1,
                "sections": [
                    {
                        "name": "other",
                        "content": "O",
                        "sections": [{"name": "rules", "content": "DUP"}],
                    }
                ],
            }
        )
    )
    (tmp_path / "combined.prompt.json").write_text(
        json.dumps(
            {
                "version": 1,
                "extends": "base.prompt.json",
                "include": ["extra.prompt.json"],
            }
        )
    )

    with pytest.raises(PromptValidationError, match="'rules' is duplicated"):
        load_prompt(tmp_path / "combined.prompt.json")


def test_cli_validate_inspect_render_nested_manifest(tmp_path: Path, capsys) -> None:
    path = tmp_path / "nested.prompt.json"
    path.write_text(json.dumps(_nested_manifest_data()))

    assert main(["prompt", "validate", str(path)]) == 0
    assert "valid prompt 'nested': 3 sections, 0 variables" in capsys.readouterr().out

    assert main(["prompt", "inspect", str(path)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["sections"][0]["sections"][0]["name"] == "rules"

    assert main(["prompt", "render", str(path)]) == 0
    assert "### rules\n\nRULES" in capsys.readouterr().out
