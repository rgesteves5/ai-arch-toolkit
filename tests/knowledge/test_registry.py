"""Tests for KnowledgeRegistry and KnowledgeEntry."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.toolkit.knowledge import (
    KnowledgeAlreadyExistsError,
    KnowledgeEntry,
    KnowledgeRegistry,
)


class TestKnowledgeEntry:
    def test_defaults(self):
        e = KnowledgeEntry(key="k", content="c")
        assert e.format == "text"
        assert e.category == ""
        assert e.tags == frozenset()
        assert e.metadata == {}
        assert e.source == ""


class TestKnowledgeRegistry:
    def test_register_and_get(self):
        reg = KnowledgeRegistry()
        entry = reg.register("rules", "no profanity")
        assert entry.key == "rules"
        assert reg.get("rules") is entry

    def test_get_missing(self):
        reg = KnowledgeRegistry()
        assert reg.get("nope") is None

    def test_require_success(self):
        reg = KnowledgeRegistry()
        reg.register("rules", "content")
        assert reg.require("rules").content == "content"

    def test_require_missing_raises(self):
        reg = KnowledgeRegistry()
        reg.register("a", "x")
        with pytest.raises(KeyError, match="Available: a"):
            reg.require("missing")

    def test_overwrite_requires_explicit_opt_in(self):
        reg = KnowledgeRegistry()
        reg.register("k", "v1")
        with pytest.raises(KnowledgeAlreadyExistsError, match="overwrite=True"):
            reg.register("k", "v2")
        reg.register("k", "v2", overwrite=True)
        assert reg.get("k").content == "v2"

    def test_remove(self):
        reg = KnowledgeRegistry()
        reg.register("k", "v")
        assert reg.remove("k")
        assert not reg.remove("k")
        assert reg.get("k") is None

    def test_clear(self):
        reg = KnowledgeRegistry()
        reg.register("a", "1")
        reg.register("b", "2")
        reg.clear()
        assert len(reg) == 0

    def test_keys_has_len_contains(self):
        reg = KnowledgeRegistry()
        reg.register("a", "1")
        reg.register("b", "2")
        assert reg.keys() == ["a", "b"]
        assert reg.has("a")
        assert not reg.has("c")
        assert len(reg) == 2
        assert "a" in reg
        assert "c" not in reg

    def test_by_category(self):
        reg = KnowledgeRegistry()
        reg.register("r1", "c1", category="rules")
        reg.register("r2", "c2", category="rules")
        reg.register("s1", "c3", category="schemas")
        assert len(reg.by_category("rules")) == 2
        assert len(reg.by_category("schemas")) == 1
        assert len(reg.by_category("other")) == 0

    def test_by_tags_match_all(self):
        reg = KnowledgeRegistry()
        reg.register("a", "x", tags=("t1", "t2"))
        reg.register("b", "y", tags=("t1",))
        result = reg.by_tags("t1", "t2", match_all=True)
        assert len(result) == 1
        assert result[0].key == "a"

    def test_by_tags_match_any(self):
        reg = KnowledgeRegistry()
        reg.register("a", "x", tags=("t1", "t2"))
        reg.register("b", "y", tags=("t3",))
        result = reg.by_tags("t1", "t3", match_all=False)
        assert len(result) == 2

    def test_search_ranks_domain_fields_and_content_deterministically(self):
        reg = KnowledgeRegistry()
        reg.register("python.rules", "Use type hints in Python code", tags=("python",))
        reg.register("general", "Python can also appear in ordinary content")
        reg.register("other", "Unrelated", category="python")

        results = reg.search("python")

        assert [result.entry.key for result in results] == [
            "python.rules",
            "other",
            "general",
        ]
        assert results[0].score > results[-1].score
        assert results[0].matched_terms == ("python",)

    def test_search_filters_and_validates_inputs(self):
        reg = KnowledgeRegistry()
        reg.register("a", "story rules", category="writing", tags=("story", "rules"))
        reg.register("b", "story style", category="writing", tags=("story",))
        reg.register("c", "story domain", category="domain", tags=("story", "rules"))
        assert [
            result.entry.key for result in reg.search("story", category="writing", tags=("rules",))
        ] == ["a"]
        assert [
            result.entry.key
            for result in reg.search("story", tags=("rules", "missing"), match_all_tags=False)
        ] == ["a", "c"]
        with pytest.raises(ValueError, match="non-empty"):
            reg.search("")
        with pytest.raises(ValueError, match="positive integer"):
            reg.search("story", limit=0)

    def test_categories_unique_sorted(self):
        reg = KnowledgeRegistry()
        reg.register("a", "x", category="z")
        reg.register("b", "y", category="a")
        reg.register("c", "z", category="z")
        assert reg.categories() == ["a", "z"]

    def test_as_context_basic(self):
        reg = KnowledgeRegistry()
        reg.register("a", "content_a")
        reg.register("b", "content_b")
        result = reg.as_context("a", "b")
        assert result == "content_a\n\n---\n\n" + "content_b"

    def test_as_context_custom_separator(self):
        reg = KnowledgeRegistry()
        reg.register("a", "A")
        reg.register("b", "B")
        assert reg.as_context("a", "b", separator="\n") == "A\nB"

    def test_as_context_custom_transform(self):
        reg = KnowledgeRegistry()
        reg.register("a", "content_a")

        def fmt(key: str, content: str) -> str:
            return f"<{key}>\n{content}\n</{key}>"

        result = reg.as_context("a", transform=fmt)
        assert result == "<a>\ncontent_a\n</a>"

    def test_as_context_missing_raises(self):
        reg = KnowledgeRegistry()
        with pytest.raises(KeyError, match="missing"):
            reg.as_context("missing")

    def test_as_context_no_keys(self):
        reg = KnowledgeRegistry()
        assert reg.as_context() == ""

    def test_register_with_source_category_tags(self):
        reg = KnowledgeRegistry()
        entry = reg.register(
            "k",
            "v",
            source="/path/to/file.txt",
            category="ref",
            tags=("tag1", "tag2"),
        )
        assert entry.source == "/path/to/file.txt"
        assert entry.category == "ref"
        assert entry.tags == frozenset({"tag1", "tag2"})
