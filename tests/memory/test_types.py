"""Tests for memory type definitions."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_arch_toolkit.toolkit.memory._types import Edge, Node, SearchResult


class TestNode:
    def test_defaults(self):
        node = Node()
        assert node.type == "generic"
        assert node.content == {}
        assert node.metadata == {}
        assert node.embedding is None
        assert node.access_count == 0
        assert node.last_accessed is None
        assert node.confidence == 1.0
        assert node.source == "unknown"
        assert len(node.id) == 16

    def test_frozen(self):
        node = Node()
        with pytest.raises(AttributeError):
            node.type = "other"  # type: ignore[misc]

    def test_lifecycle_fields(self):
        node = Node(access_count=5, confidence=0.8, source="user_stated")
        assert node.access_count == 5
        assert node.confidence == 0.8
        assert node.source == "user_stated"

    def test_bi_temporal(self):
        now = datetime.now(UTC)
        node = Node()
        assert node.timestamp >= now or (now - node.timestamp).total_seconds() < 1
        assert node.created_at >= now or (now - node.created_at).total_seconds() < 1

    def test_source_values(self):
        for src in ("user_stated", "agent_inferred", "tool_result", "external", "consolidated"):
            node = Node(source=src)
            assert node.source == src

    def test_custom_content(self):
        node = Node(content={"text": "hello", "subject": "greeting"})
        assert node.content["text"] == "hello"


class TestEdge:
    def test_defaults(self):
        edge = Edge(source="a", target="b", relation="KNOWS")
        assert edge.weight == 1.0
        assert edge.metadata == {}

    def test_custom(self):
        edge = Edge(source="a", target="b", relation="LIKES", weight=0.5)
        assert edge.weight == 0.5


class TestSearchResult:
    def test_creation(self):
        node = Node(content={"text": "test"})
        result = SearchResult(node=node, score=0.95)
        assert result.score == 0.95
        assert result.node.content["text"] == "test"
