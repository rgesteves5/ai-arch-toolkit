"""Failure-path and less common prompt-manifest coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_arch_toolkit.toolkit.knowledge import KnowledgeRegistry
from ai_arch_toolkit.toolkit.prompts import PromptLoadError, PromptValidationError, load_prompt


def manifest(tmp_path: Path, payload: object, name: str = "test.prompt.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


@pytest.mark.parametrize("name", ["bad.prompt.csv", "almost.prompt.json.bak", "prompt.yaml"])
def test_manifest_rejects_unsupported_or_ambiguous_filenames(tmp_path: Path, name: str) -> None:
    path = manifest(tmp_path, {"version": 1}, name)
    with pytest.raises(PromptLoadError, match="prompt manifests must use"):
        load_prompt(path)


def test_manifest_rejects_invalid_depth_and_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.prompt.json"
    with pytest.raises(ValueError, match="at least 1"):
        load_prompt(path, max_include_depth=0)
    with pytest.raises(PromptLoadError, match="could not load prompt manifest"):
        load_prompt(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("extends", [], "extends must be a non-empty"),
        ("include", 1, "include must be a path"),
        ("include", [""], "include paths must be non-empty"),
        ("variables", [], "variables must be an object"),
        ("separator", 1, "separator must be a string"),
        ("name", 1, "name and description must be strings"),
        ("description", [], "name and description must be strings"),
        ("metadata", [], "metadata must be an object"),
    ],
)
def test_invalid_top_level_values_are_contextual(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    path = manifest(tmp_path, {"version": 1, field: value})
    with pytest.raises(PromptValidationError, match=message):
        load_prompt(path)


@pytest.mark.parametrize(
    ("variables", "message"),
    [
        ({"value": 7}, "object or type name"),
        ({"value": {"type": "date"}}, "invalid prompt variable"),
        ({"value": {"required": "yes"}}, "invalid prompt variable"),
        ({"value": {"descripton": "typo"}}, "did you mean 'description'"),
    ],
)
def test_invalid_variable_declarations(tmp_path: Path, variables: object, message: str) -> None:
    path = manifest(tmp_path, {"version": 1, "variables": variables})
    with pytest.raises(PromptValidationError, match=message):
        load_prompt(path)


def test_variable_type_shorthand_and_default_are_supported(tmp_path: Path) -> None:
    path = manifest(
        tmp_path,
        {
            "version": 1,
            "variables": {"count": "integer", "topic": {"default": "graphs"}},
            "sections": [
                {
                    "name": "request",
                    "template": {"content": "${topic}: ${count}"},
                }
            ],
        },
    )
    assert load_prompt(path).render(count=2).text == "graphs: 2"


@pytest.mark.parametrize(
    ("section", "message"),
    [
        (1, "at index 0 must be an object"),
        ({"content": "x"}, "requires a name"),
        ({"name": "x", "content": "x", "remove": 1}, "flags must be booleans"),
        ({"name": "x"}, "exactly one"),
        ({"name": "x", "content": 1}, "content must be a string"),
        ({"name": "x", "content": "x", "order": "first"}, "invalid prompt section"),
        ({"name": "x", "content": "x", "stability": "daily"}, "invalid prompt section"),
        ({"name": "x", "content": "x", "metadata": []}, "metadata must be an object"),
        ({"name": "x", "content": "x", "oder": 1}, "did you mean 'order'"),
    ],
)
def test_invalid_section_declarations(tmp_path: Path, section: object, message: str) -> None:
    path = manifest(tmp_path, {"version": 1, "sections": [section]})
    with pytest.raises(PromptValidationError, match=message):
        load_prompt(path)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (1, "source must be a path or object"),
        ({}, "source requires a non-empty path"),
        ({"path": "data.json", "selector": "/x"}, "unknown fields"),
        ({"path": "data.json", "select": 7}, "selector must be a string or object"),
        ({"path": "data.json", "select": {"type": "csv"}}, "unknown resource selector"),
        (
            {"path": "data.json", "select": {"type": "json_pointer", "value": 1}},
            "invalid prompt section",
        ),
        (
            {"path": "data.json", "select": {"type": "lines", "start": -1}},
            "invalid prompt section",
        ),
        (
            {"path": "data.json", "select": {"type": "block", "start_marker": ""}},
            "invalid prompt section",
        ),
    ],
)
def test_invalid_resource_sources(tmp_path: Path, source: object, message: str) -> None:
    (tmp_path / "data.json").write_text('{"x": 1}')
    path = manifest(
        tmp_path,
        {"version": 1, "sections": [{"name": "x", "source": source}]},
    )
    with pytest.raises(PromptValidationError, match=message):
        load_prompt(path)


@pytest.mark.parametrize(
    ("template", "message"),
    [
        (1, "template must be a path or object"),
        ({}, "exactly one of path or content"),
        ({"path": "x.md", "content": "x"}, "exactly one of path or content"),
        ({"content": 1}, "content must be a string"),
        ({"content": "x", "engine": 1}, "engine must be a string"),
        ({"content": "x", "engine": "unknown"}, "unknown template engine"),
    ],
)
def test_invalid_template_sources(tmp_path: Path, template: object, message: str) -> None:
    path = manifest(
        tmp_path,
        {"version": 1, "sections": [{"name": "x", "template": template}]},
    )
    with pytest.raises(PromptValidationError, match=message):
        load_prompt(path)


@pytest.mark.parametrize(
    ("knowledge", "message"),
    [
        (1, "key, key list, or object"),
        ({}, "keys must be a list"),
        ({"keys": [""]}, "invalid prompt section"),
        ({"keys": ["x"], "separator": 1}, "invalid prompt section"),
        ({"keys": ["x"], "include_names": 1}, "invalid prompt section"),
        ({"keys": ["x"], "names": True}, "unknown fields"),
    ],
)
def test_invalid_knowledge_sources(tmp_path: Path, knowledge: object, message: str) -> None:
    registry = KnowledgeRegistry()
    path = manifest(
        tmp_path,
        {"version": 1, "sections": [{"name": "x", "knowledge": knowledge}]},
    )
    with pytest.raises(PromptValidationError, match=message):
        load_prompt(path, knowledge=registry)


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        ("csv", "unknown prompt layout"),
        (1, "layout must be a name or object"),
        ({}, "requires a type"),
        ({"type": "csv"}, "unknown prompt layout type"),
        ({"type": "text", "separator": 1}, "invalid prompt layout"),
        ({"type": "text", "between": {}}, "between must be a list"),
        ({"type": "text", "between": [1]}, "boundary must be an object"),
        (
            {"type": "text", "between": [{"from": "a", "to": "b", "separator": 1}]},
            "from, to, and separator must be strings",
        ),
        ({"type": "markdown", "heading_level": 7}, "invalid prompt layout"),
        ({"type": "xml", "root_tag": "bad tag"}, "invalid prompt layout"),
        ({"type": "json", "indent": "two"}, "invalid prompt layout"),
    ],
)
def test_invalid_layouts(tmp_path: Path, layout: object, message: str) -> None:
    path = manifest(tmp_path, {"version": 1, "layout": layout})
    with pytest.raises(PromptValidationError, match=message):
        load_prompt(path)


def test_duplicate_sections_and_variables_from_includes_are_rejected(tmp_path: Path) -> None:
    shared = {
        "version": 1,
        "variables": {"topic": "string"},
        "sections": [{"name": "rules", "content": "rules"}],
    }
    manifest(tmp_path, shared, "one.prompt.json")
    manifest(tmp_path, shared, "two.prompt.json")
    duplicate_variable = manifest(
        tmp_path,
        {"version": 1, "include": ["one.prompt.json", "two.prompt.json"]},
        "variables.prompt.json",
    )
    with pytest.raises(PromptValidationError, match=r"variable 'topic'.*duplicated"):
        load_prompt(duplicate_variable)

    second = dict(shared)
    second["variables"] = {"other": "string"}
    manifest(tmp_path, second, "two.prompt.json")
    duplicate_section = manifest(
        tmp_path,
        {"version": 1, "include": ["one.prompt.json", "two.prompt.json"]},
        "sections.prompt.json",
    )
    with pytest.raises(PromptValidationError, match=r"section 'rules'.*duplicated"):
        load_prompt(duplicate_section)


def test_local_duplicate_section_requires_replace(tmp_path: Path) -> None:
    path = manifest(
        tmp_path,
        {
            "version": 1,
            "sections": [
                {"name": "x", "content": "one"},
                {"name": "x", "content": "two"},
            ],
        },
    )
    with pytest.raises(PromptValidationError, match=r"section 'x'.*duplicated"):
        load_prompt(path)


def test_less_common_selectors_layouts_and_knowledge_shorthands(tmp_path: Path) -> None:
    (tmp_path / "data.json").write_text('{"a/b": {"~key": "VALUE"}}')
    (tmp_path / "blocks.txt").write_text("before\nBEGIN\ninside\nEND\nafter")
    registry = KnowledgeRegistry()
    registry.register("first", "ONE")
    registry.register("second", "TWO")
    path = manifest(
        tmp_path,
        {
            "version": 1,
            "layout": {
                "type": "markdown",
                "heading_level": 3,
                "include_headings": False,
                "separator": "|",
            },
            "sections": [
                {
                    "name": "pointer",
                    "source": {
                        "path": "data.json",
                        "select": {"type": "json_pointer", "value": "/a~1b/~0key"},
                    },
                },
                {
                    "name": "block",
                    "order": 1,
                    "source": {
                        "path": "blocks.txt",
                        "select": {
                            "type": "block",
                            "start_marker": "BEGIN",
                            "end_marker": "END",
                            "include_markers": True,
                        },
                    },
                },
                {"name": "knowledge", "order": 2, "knowledge": ["first", "second"]},
            ],
        },
    )
    rendered = load_prompt(path, knowledge=registry).render()
    assert rendered.layout == "markdown"
    assert rendered.text == "VALUE|BEGIN\ninside\nEND\n|ONE\n\n---\n\nTWO"


def test_knowledge_string_shorthand_and_null_layout(tmp_path: Path) -> None:
    registry = KnowledgeRegistry()
    registry.register("rules", "RULES")
    path = manifest(
        tmp_path,
        {
            "version": 1,
            "layout": None,
            "sections": [{"name": "knowledge", "knowledge": "rules"}],
        },
    )
    assert load_prompt(path, knowledge=registry).render().text == "RULES"


@pytest.mark.parametrize("layout", [{"type": "json"}, {"type": "xml"}])
def test_object_json_and_xml_layouts_render(tmp_path: Path, layout: object) -> None:
    path = manifest(
        tmp_path,
        {"version": 1, "layout": layout, "sections": [{"name": "x", "content": "X"}]},
    )
    rendered = load_prompt(path).render()
    assert rendered.sections[0].content == "X"
    assert "X" in rendered.section_text("x")
