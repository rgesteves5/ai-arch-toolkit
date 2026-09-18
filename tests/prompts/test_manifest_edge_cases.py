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
        ("include", 1, "include must be a non-empty string or a list"),
        ("include", [""], r"include\[0\] must be a non-empty string"),
        ("variables", [], "variables must be an object"),
        ("separator", 1, "separator must be a string"),
        ("name", 1, "name must be a string"),
        ("description", [], "description must be a string"),
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
        ({"value": 7}, r"variables\.value must be one of 'any', .* or an object"),
        ({"value": {"type": "date"}}, r"variables\.value\.type must be one of 'any'"),
        ({"value": {"required": "yes"}}, r"variables\.value\.required must be a boolean"),
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
        (1, r"sections\[0\] must be an object"),
        ({"content": "x"}, r"sections\[0\] is missing 'name'"),
        ({"name": "x", "content": "x", "remove": 1}, r"sections\[0\]\.remove must be a boolean"),
        ({"name": "x"}, "exactly one"),
        ({"name": "x", "content": 1}, r"sections\[0\]\.content must be a string"),
        (
            {"name": "x", "content": "x", "order": "first"},
            r"sections\[0\]\.order must be an integer",
        ),
        (
            {"name": "x", "content": "x", "stability": "daily"},
            r"sections\[0\]\.stability must be one of 'static', 'session', 'request'",
        ),
        (
            {"name": "x", "content": "x", "metadata": []},
            r"sections\[0\]\.metadata must be an object",
        ),
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
        (1, r"sections\[0\]\.source must be a non-empty string or an object"),
        ({}, r"sections\[0\]\.source is missing 'path'"),
        (
            {"path": "data.json", "selector": "/x"},
            r"source has unknown fields: 'selector' \(did you mean 'select'\?\)",
        ),
        ({"path": "data.json", "select": 7}, r"source\.select must be a string or an object"),
        (
            {"path": "data.json", "select": {"type": "csv"}},
            r"select\.type must be one of 'json_pointer', 'heading', 'lines', 'block'",
        ),
        (
            {"path": "data.json", "select": {"type": "json_pointer", "value": 1}},
            r"select\.value must be a string",
        ),
        (
            {"path": "data.json", "select": {"type": "lines", "start": -1}},
            r"select\.start must be a positive integer",
        ),
        (
            {"path": "data.json", "select": {"type": "block", "start_marker": ""}},
            r"select is missing 'end_marker'",
        ),
        (
            {
                "path": "data.json",
                "select": {"type": "block", "start_marker": "", "end_marker": "x"},
            },
            r"select\.start_marker must be a non-empty string",
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
        (1, r"sections\[0\]\.template must be a non-empty string or an object"),
        ({}, "exactly one of path or content"),
        ({"path": "x.md", "content": "x"}, "exactly one of path or content"),
        ({"content": 1}, r"template\.content must be a string"),
        ({"content": "x", "engine": 1}, r"template\.engine must be a non-empty string"),
        ({"content": "x", "engine": "unknown"}, "unknown template engine"),
        ({"content": "x", "select": "/a"}, "inline prompt template content cannot use select"),
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
        (1, r"sections\[0\]\.knowledge must be a non-empty string, a list or an object"),
        ({}, r"knowledge is missing 'keys'"),
        ({"keys": [""]}, r"knowledge\.keys\[0\] must be a non-empty string"),
        ({"keys": ["x"], "separator": 1}, r"knowledge\.separator must be a string"),
        ({"keys": ["x"], "include_names": 1}, r"knowledge\.include_names must be a boolean"),
        ({"keys": ["x"], "names": True}, r"knowledge has unknown fields: 'names'"),
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
        ("csv", "layout must be one of 'text', 'markdown', 'xml', 'json'"),
        (1, "layout must be one of 'text', 'markdown', 'xml', 'json' or an object"),
        ({}, "layout is missing 'type'"),
        ({"type": "csv"}, r"layout\.type must be one of 'text', 'markdown', 'xml', 'json'"),
        ({"type": "text", "separator": 1}, r"layout\.separator must be a string"),
        ({"type": "text", "between": {}}, r"layout\.between must be a list"),
        ({"type": "text", "between": [1]}, r"layout\.between\[0\] must be an object"),
        (
            {"type": "text", "between": [{"from": "a", "to": "b", "separator": 1}]},
            r"layout\.between\[0\]\.separator must be a string",
        ),
        (
            {"type": "markdown", "heading_level": 7},
            r"layout\.heading_level must be an integer between 1 and 6",
        ),
        (
            {"type": "xml", "root_tag": "bad tag"},
            r"layout\.root_tag must be a non-empty string matching",
        ),
        ({"type": "json", "indent": "two"}, r"layout\.indent must be a non-negative integer"),
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
