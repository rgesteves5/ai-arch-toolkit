"""Tests for knowledge file loaders."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from ai_arch_toolkit.toolkit.knowledge import (
    KnowledgeRegistry,
    load_directory,
    load_json,
    load_markdown,
    load_text,
    load_toml,
    load_yaml,
)


@pytest.fixture
def reg():
    return KnowledgeRegistry()


class TestLoadText:
    def test_load(self, reg, tmp_path):
        p = tmp_path / "file.txt"
        p.write_text("hello world")
        entry = load_text(reg, "greeting", p)
        assert entry.content == "hello world"
        assert entry.format == "text"
        assert entry.source == str(p)


class TestLoadJson:
    def test_load_and_format(self, reg, tmp_path):
        p = tmp_path / "data.json"
        p.write_text('{"a":1,"b":2}')
        entry = load_json(reg, "data", p)
        parsed = json.loads(entry.content)
        assert parsed == {"a": 1, "b": 2}
        assert entry.format == "json"
        # Should be formatted (indented)
        assert "  " in entry.content

    def test_invalid_json_raises(self, reg, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{invalid")
        with pytest.raises(json.JSONDecodeError):
            load_json(reg, "bad", p)


class TestLoadToml:
    def test_load_and_validate(self, reg, tmp_path):
        p = tmp_path / "config.toml"
        content = '[section]\nkey = "value"\n'
        p.write_text(content)
        entry = load_toml(reg, "config", p)
        assert entry.content == content
        assert entry.format == "toml"


class TestLoadMarkdown:
    def test_load(self, reg, tmp_path):
        p = tmp_path / "doc.md"
        p.write_text("# Title\n\nBody text")
        entry = load_markdown(reg, "doc", p)
        assert entry.content == "# Title\n\nBody text"
        assert entry.format == "markdown"


class TestLoadYaml:
    def test_load_with_pyyaml(self, reg, tmp_path):
        p = tmp_path / "data.yaml"
        p.write_text("key: value\n")
        entry = load_yaml(reg, "data", p)
        assert entry.content == "key: value\n"
        assert entry.format == "yaml"

    def test_without_pyyaml_raises(self, reg, tmp_path):
        p = tmp_path / "data.yaml"
        p.write_text("key: value\n")
        with patch.dict("sys.modules", {"yaml": None}), pytest.raises(ImportError, match="pyyaml"):
            load_yaml(reg, "data", p)


class TestLoadDirectory:
    def test_basic(self, reg, tmp_path):
        (tmp_path / "a.txt").write_text("A")
        (tmp_path / "b.json").write_text('{"x": 1}')
        count = load_directory(reg, tmp_path)
        assert count == 2
        assert reg.has("a")
        assert reg.has("b")

    def test_with_prefix(self, reg, tmp_path):
        (tmp_path / "a.txt").write_text("A")
        load_directory(reg, tmp_path, prefix="ref.")
        assert reg.has("ref.a")

    def test_with_extensions_filter(self, reg, tmp_path):
        (tmp_path / "a.txt").write_text("A")
        (tmp_path / "b.json").write_text('{"x": 1}')
        count = load_directory(reg, tmp_path, extensions={".txt"})
        assert count == 1
        assert reg.has("a")
        assert not reg.has("b")

    def test_recursive(self, reg, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.txt").write_text("R")
        (sub / "nested.txt").write_text("N")
        count = load_directory(reg, tmp_path, recursive=True)
        assert count == 2
        assert reg.has("root")
        assert reg.has("sub.nested")

    def test_stem_collision_raises(self, reg, tmp_path):
        (tmp_path / "data.txt").write_text("A")
        (tmp_path / "data.json").write_text('{"x": 1}')
        with pytest.raises(ValueError, match="collision"):
            load_directory(reg, tmp_path)

    def test_sorted_deterministic(self, reg, tmp_path):
        (tmp_path / "z.txt").write_text("Z")
        (tmp_path / "a.txt").write_text("A")
        (tmp_path / "m.txt").write_text("M")
        load_directory(reg, tmp_path)
        assert reg.keys() == ["a", "m", "z"]

    def test_skips_unknown_extensions(self, reg, tmp_path):
        (tmp_path / "a.txt").write_text("A")
        (tmp_path / "b.xyz").write_text("X")
        count = load_directory(reg, tmp_path)
        assert count == 1

    def test_recursive_with_prefix(self, reg, tmp_path):
        sub = tmp_path / "deep"
        sub.mkdir()
        (sub / "config.json").write_text('{"k": 1}')
        load_directory(reg, tmp_path, recursive=True, prefix="kb.")
        assert reg.has("kb.deep.config")
