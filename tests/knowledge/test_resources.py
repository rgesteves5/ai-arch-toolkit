"""Knowledge integration with reusable resources and prompts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ai_arch_toolkit.toolkit.knowledge import KnowledgeEntry, KnowledgeRegistry
from ai_arch_toolkit.toolkit.prompts import Prompt, PromptSection
from ai_arch_toolkit.toolkit.resources import Resource, ResourceResolver, load_resource


def test_literal_entry_exposes_fingerprint_media_type_and_data() -> None:
    entry = KnowledgeEntry(key="rules", content="Be concise.")

    assert entry.data == "Be concise."
    assert entry.media_type == "text/plain"
    assert entry.fingerprint == "sha256:" + hashlib.sha256(b"Be concise.").hexdigest()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"key": "", "content": "x"}, "key is required"),
        ({"key": "x", "content": 1}, "content must be a string"),
        ({"key": "x", "content": "x", "format": 1}, "format must be a string"),
        ({"key": "x", "content": "x", "category": 1}, "category must be a string"),
        ({"key": "x", "content": "x", "source": 1}, "source must be a string"),
        ({"key": "x", "content": "x", "tags": "tag"}, "tags must contain strings"),
        ({"key": "x", "content": "x", "tags": 1}, "tags must contain strings"),
        ({"key": "x", "content": "x", "tags": (1,)}, "tags must contain strings"),
        ({"key": "x", "content": "x", "metadata": []}, "metadata must be a mapping"),
        ({"key": "x", "content": "x", "resource": "bad"}, "resource must be"),
    ],
)
def test_knowledge_entry_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        KnowledgeEntry(**kwargs)  # type: ignore[arg-type]


def test_register_resource_preserves_parsed_data_and_provenance(tmp_path: Path) -> None:
    path = tmp_path / "knowledge.json"
    path.write_text('{"writing": {"rules": ["short", "clear"]}}')
    resource = load_resource(path)
    registry = KnowledgeRegistry()

    entry = registry.register_resource(
        "writing.rules",
        resource,
        selector="/writing/rules",
        serialize_as="markdown",
        tags=("writing",),
    )

    assert entry.content == "- short\n- clear"
    assert entry.data == {"writing": {"rules": ["short", "clear"]}}
    assert entry.media_type == "application/json"
    assert entry.fingerprint == resource.fingerprint
    assert entry.source == str(path)


def test_registry_load_preserves_whole_source_by_default(tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text("rules:\n  - concise\n")
    registry = KnowledgeRegistry()

    entry = registry.load("rules", path)

    assert entry.content == "rules:\n  - concise\n"
    assert entry.format == "yaml"
    assert entry.data == {"rules": ["concise"]}


def test_registry_from_directory_generates_deterministic_nested_keys(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "b" / "rules.txt").write_text("B")
    (tmp_path / "a" / "rules.txt").write_text("A")

    registry = KnowledgeRegistry.from_directory(tmp_path, recursive=True, prefix="kb.")

    assert registry.keys() == ["kb.a.rules", "kb.b.rules"]


def test_prompt_section_from_knowledge_removes_manual_context_plumbing() -> None:
    registry = KnowledgeRegistry()
    registry.register("style", "Be concise.")
    registry.register("domain", "Use architecture terms.")

    section = PromptSection.from_knowledge(
        registry,
        ["style", "domain"],
        include_names=True,
        order=200,
    )
    rendered = Prompt(
        sections=(PromptSection(name="role", content="Architect", order=100), section)
    ).render()

    assert rendered.text == (
        "Architect\n\n[style]\nBe concise.\n\n---\n\n[domain]\nUse architecture terms."
    )
    assert section.metadata["source_provenance"]["keys"] == ("style", "domain")


def test_register_resource_binary_requires_explicit_supported_serialization() -> None:
    registry = KnowledgeRegistry()
    resource = Resource(
        ref=Resource.from_text("x").ref,
        raw=b"\x00",
        media_type="application/octet-stream",
        data=b"\x00",
    )
    with pytest.raises(ValueError, match="binary"):
        registry.register_resource("binary", resource)
    with pytest.raises(TypeError, match="must be a Resource"):
        registry.register_resource("bad", object())  # type: ignore[arg-type]


def test_registry_load_uses_custom_serializer_registered_on_resolver(tmp_path: Path) -> None:
    class PrefixSerializer:
        name = "prefix"

        def serialize(self, value: object) -> str:
            return f"prefix:{value}"

    path = tmp_path / "knowledge.json"
    path.write_text('{"rules": ["short"]}')
    resolver = ResourceResolver()
    resolver.register_serializer("prefix", PrefixSerializer())

    entry = KnowledgeRegistry().load(
        "rules",
        path,
        selector="/rules",
        serialize_as="prefix",
        resolver=resolver,
    )

    assert entry.content == "prefix:['short']"
