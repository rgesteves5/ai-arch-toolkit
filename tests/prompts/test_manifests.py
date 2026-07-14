"""Declarative prompt-manifest tests."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from ai_arch_toolkit.toolkit.knowledge import KnowledgeRegistry
from ai_arch_toolkit.toolkit.prompts import (
    PromptIncludeCycleError,
    PromptLoadError,
    PromptTemplate,
    PromptValidationError,
    load_prompt,
)


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_yaml_manifest_end_to_end(tmp_path: Path) -> None:
    write(tmp_path / "role.md", "You are a ${role}.")
    write(
        tmp_path / "rules.yaml",
        "genres:\n  mystery:\n    - Plant clues\n    - Keep suspense\n",
    )
    write(tmp_path / "request.template.md", "Write ${task} for ${audience}.")
    manifest = write(
        tmp_path / "writer.prompt.yaml",
        """
version: 1
name: story-writer
description: Writes stories.
layout:
  type: xml
  root_tag: instructions
variables:
  role:
    type: string
    required: true
  genre:
    type: string
    required: true
  audience:
    type: string
    default: general readers
sections:
  - name: role
    template: role.md
    order: 100
  - name: rules
    source:
      path: rules.yaml
      select: /genres/${genre}
      serialize_as: markdown
    order: 200
  - name: request
    template:
      path: request.template.md
      engine: string-template
    order: 900
    stability: request
""".lstrip(),
    )

    template = load_prompt(manifest)
    rendered = template.render(
        role="story architect",
        genre="mystery",
        task="chapter one",
    )

    assert template.name == "story-writer"
    assert template.variable_names == ("role", "genre", "audience", "task")
    assert rendered.layout == "xml"
    assert "You are a story architect." in rendered.text
    assert "- Plant clues\n- Keep suspense" in rendered.text
    assert "Write chapter one for general readers." in rendered.text
    assert rendered.provenance["metadata"]["manifest"] == str(manifest.resolve())


@pytest.mark.parametrize("suffix", ["json", "toml"])
def test_json_and_toml_manifests(tmp_path: Path, suffix: str) -> None:
    if suffix == "json":
        content = json.dumps(
            {
                "version": 1,
                "name": "inline",
                "layout": "text",
                "variables": {"name": {"required": True}},
                "sections": [
                    {
                        "name": "greeting",
                        "template": {"content": "Hello ${name}"},
                    }
                ],
            }
        )
    else:
        content = """
version = 1
name = "inline"
layout = "text"

[variables.name]
required = true

[[sections]]
name = "greeting"

[sections.template]
content = "Hello ${name}"
""".lstrip()
    path = write(tmp_path / f"inline.prompt.{suffix}", content)

    template = load_prompt(path)

    assert template.render(name="Ada").text == "Hello Ada"


def test_manifest_infers_inline_template_variables(tmp_path: Path) -> None:
    path = write(
        tmp_path / "inferred.prompt.json",
        json.dumps(
            {
                "version": 1,
                "sections": [{"name": "request", "template": {"content": "${task}: ${topic}"}}],
            }
        ),
    )

    template = load_prompt(path)

    assert template.variable_names == ("task", "topic")
    assert template.render(task="Explain", topic="graphs").text == "Explain: graphs"


def test_manifest_knowledge_source(tmp_path: Path) -> None:
    registry = KnowledgeRegistry()
    registry.register("style", "Be concise.")
    path = write(
        tmp_path / "knowledge.prompt.json",
        json.dumps(
            {
                "version": 1,
                "sections": [
                    {
                        "name": "knowledge",
                        "knowledge": {"keys": ["style"], "include_names": True},
                    }
                ],
            }
        ),
    )

    with pytest.raises(PromptValidationError, match="received no registry"):
        load_prompt(path)
    assert load_prompt(path, knowledge=registry).render().text == "[style]\nBe concise."


def test_manifest_explicit_selectors(tmp_path: Path) -> None:
    write(tmp_path / "guide.md", "# Intro\nintro\n# Rules\nA\nB\n# End\nend\n")
    path = write(
        tmp_path / "selectors.prompt.json",
        json.dumps(
            {
                "version": 1,
                "sections": [
                    {
                        "name": "heading",
                        "source": {
                            "path": "guide.md",
                            "select": {"type": "heading", "heading": "Rules"},
                        },
                        "order": 1,
                    },
                    {
                        "name": "lines",
                        "source": {
                            "path": "guide.md",
                            "select": {"type": "lines", "start": 2, "end": 2},
                        },
                        "order": 2,
                    },
                ],
            }
        ),
    )

    rendered = load_prompt(path).render()

    assert rendered.text == "A\nB\n\nintro\n"


def test_text_layout_boundary_configuration(tmp_path: Path) -> None:
    path = write(
        tmp_path / "layout.prompt.json",
        json.dumps(
            {
                "version": 1,
                "layout": {
                    "type": "text",
                    "separator": "|",
                    "between": [{"from": "role", "to": "request", "separator": "\nREQUEST\n"}],
                },
                "sections": [
                    {"name": "role", "content": "ROLE"},
                    {"name": "request", "content": "TASK", "order": 1},
                ],
            }
        ),
    )

    assert load_prompt(path).render().text == "ROLE\nREQUEST\nTASK"


def test_include_combines_sections(tmp_path: Path) -> None:
    write(
        tmp_path / "shared.prompt.json",
        json.dumps(
            {
                "version": 1,
                "sections": [{"name": "rules", "content": "RULES", "order": 100}],
            }
        ),
    )
    path = write(
        tmp_path / "main.prompt.json",
        json.dumps(
            {
                "version": 1,
                "include": "shared.prompt.json",
                "sections": [{"name": "request", "content": "TASK", "order": 200}],
            }
        ),
    )

    assert load_prompt(path).render().text == "RULES\n\nTASK"


def test_extends_replace_remove_and_add(tmp_path: Path) -> None:
    write(
        tmp_path / "base.prompt.json",
        json.dumps(
            {
                "version": 1,
                "name": "base",
                "sections": [
                    {"name": "role", "content": "OLD", "order": 100},
                    {"name": "unused", "content": "REMOVE", "order": 200},
                ],
            }
        ),
    )
    child = write(
        tmp_path / "child.prompt.json",
        json.dumps(
            {
                "version": 1,
                "extends": "base.prompt.json",
                "name": "child",
                "sections": [
                    {"name": "role", "content": "NEW", "replace": True, "order": 100},
                    {"name": "unused", "remove": True},
                    {"name": "request", "content": "TASK", "order": 200},
                ],
            }
        ),
    )

    template = load_prompt(child)

    assert template.name == "child"
    assert template.render().text == "NEW\n\nTASK"


def test_include_and_extends_cycles_are_detected(tmp_path: Path) -> None:
    a = write(
        tmp_path / "a.prompt.json",
        json.dumps({"version": 1, "include": "b.prompt.json"}),
    )
    write(
        tmp_path / "b.prompt.json",
        json.dumps({"version": 1, "extends": "a.prompt.json"}),
    )

    with pytest.raises(PromptIncludeCycleError, match=r"a\.prompt\.json.*b\.prompt\.json"):
        load_prompt(a)


def test_include_depth_is_limited(tmp_path: Path) -> None:
    write(tmp_path / "c.prompt.json", json.dumps({"version": 1}))
    write(
        tmp_path / "b.prompt.json",
        json.dumps({"version": 1, "include": "c.prompt.json"}),
    )
    a = write(
        tmp_path / "a.prompt.json",
        json.dumps({"version": 1, "include": "b.prompt.json"}),
    )

    with pytest.raises(PromptValidationError, match="include depth"):
        load_prompt(a, max_include_depth=2)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "version: 1"),
        ({"version": 2}, "version: 1"),
        ({"version": 1, "sectons": []}, "did you mean 'sections'"),
        ({"version": 1, "sections": {}}, "sections must be a list"),
        (
            {"version": 1, "sections": [{"name": "x", "content": "a", "source": "b"}]},
            "exactly one",
        ),
    ],
)
def test_invalid_manifest_errors_are_contextual(
    tmp_path: Path, payload: object, message: str
) -> None:
    path = write(tmp_path / "bad.prompt.json", json.dumps(payload))

    with pytest.raises(PromptValidationError, match=message):
        load_prompt(path)


def test_manifest_requires_explicit_filename(tmp_path: Path) -> None:
    path = write(tmp_path / "prompt.json", json.dumps({"version": 1}))
    with pytest.raises(PromptLoadError, match=r"\.prompt\.json"):
        load_prompt(path)


def test_manifest_top_level_must_be_object(tmp_path: Path) -> None:
    path = write(tmp_path / "list.prompt.json", "[]")
    with pytest.raises(PromptValidationError, match="must contain an object"):
        load_prompt(path)


def test_manifest_path_cannot_escape_default_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    write(tmp_path / "outside.md", "secret")
    path = write(
        root / "bad.prompt.json",
        json.dumps(
            {
                "version": 1,
                "sections": [{"name": "x", "source": "../outside.md"}],
            }
        ),
    )

    with pytest.raises(PromptLoadError, match="outside allowed roots"):
        load_prompt(path)


def test_invalid_section_operations(tmp_path: Path) -> None:
    base = write(
        tmp_path / "base.prompt.json",
        json.dumps({"version": 1, "sections": [{"name": "x", "content": "x"}]}),
    )
    payloads = [
        ({"name": "x", "remove": True, "replace": True}, "both remove and replace"),
        ({"name": "x", "remove": True, "content": "x"}, "cannot define content"),
        ({"name": "missing", "remove": True}, "cannot remove unknown"),
        ({"name": "missing", "content": "x", "replace": True}, "cannot replace unknown"),
    ]
    for index, (section, message) in enumerate(payloads):
        path = write(
            tmp_path / f"invalid-{index}.prompt.json",
            json.dumps({"version": 1, "extends": base.name, "sections": [section]}),
        )
        with pytest.raises(PromptValidationError, match=message):
            load_prompt(path)


def test_prompt_template_from_manifest_alias(tmp_path: Path) -> None:
    path = write(
        tmp_path / "alias.prompt.json",
        json.dumps({"version": 1, "sections": [{"name": "x", "content": "X"}]}),
    )
    assert PromptTemplate.from_manifest(path).render().text == "X"


def test_manifest_schema_is_packaged_and_valid_json() -> None:
    schema_path = (
        Path(__file__).parents[2]
        / "src/ai_arch_toolkit/toolkit/prompts/schemas/prompt-manifest-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    assert schema["properties"]["version"] == {"const": 1}


def test_package_manifest_supports_relative_sources_and_includes(
    tmp_path: Path, monkeypatch
) -> None:
    package = tmp_path / "prompt_package"
    package.mkdir()
    write(package / "__init__.py", "")
    write(package / "role.md", "PACKAGE ROLE")
    write(
        package / "shared.prompt.json",
        json.dumps({"version": 1, "sections": [{"name": "shared", "content": "SHARED"}]}),
    )
    write(
        package / "main.prompt.json",
        json.dumps(
            {
                "version": 1,
                "include": "shared.prompt.json",
                "sections": [{"name": "role", "source": "role.md", "order": 1}],
            }
        ),
    )
    monkeypatch.syspath_prepend(tmp_path)
    importlib.invalidate_caches()

    template = load_prompt("package://prompt_package/main.prompt.json")

    assert template.render().text == "SHARED\n\nPACKAGE ROLE"


def test_manifest_uses_custom_serializer_registered_on_resolver(tmp_path: Path) -> None:
    from ai_arch_toolkit.toolkit.resources import ResourceResolver

    class UpperSerializer:
        name = "upper"

        def serialize(self, value):
            return str(value).upper()

    write(tmp_path / "value.json", json.dumps({"name": "ada"}))
    path = write(
        tmp_path / "custom.prompt.json",
        json.dumps(
            {
                "version": 1,
                "sections": [
                    {
                        "name": "value",
                        "source": {
                            "path": "value.json",
                            "select": "/name",
                            "serialize_as": "upper",
                        },
                    }
                ],
            }
        ),
    )
    resolver = ResourceResolver()
    resolver.register_serializer("upper", UpperSerializer())
    assert load_prompt(path, resolver=resolver).render().text == "ADA"
