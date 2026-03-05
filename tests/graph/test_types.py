"""Tests for core graph data types."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ai_arch_toolkit.core.graph._types import Edge, Node


class TestNode:
    def test_defaults(self):
        node: Node[None] = Node()
        assert node.type == "default"
        assert node.content is None
        assert node.metadata == {}
        assert len(node.id) == 16

    def test_string_content(self):
        node: Node[str] = Node(content="hello")
        assert node.content == "hello"

    def test_dict_content(self):
        node: Node[dict] = Node(content={"name": "Alice", "age": 30})
        assert node.content["name"] == "Alice"

    def test_dataclass_content(self):
        @dataclass(frozen=True, slots=True)
        class Character:
            name: str
            role: str

        node: Node[Character] = Node(content=Character(name="Alice", role="hero"))
        assert node.content.name == "Alice"
        assert node.content.role == "hero"

    def test_frozen(self):
        node: Node[str] = Node(content="hello")
        with pytest.raises(AttributeError):
            node.type = "other"  # type: ignore[misc]

    def test_custom_id_and_type(self):
        node: Node[str] = Node(id="custom-id", type="character", content="Alice")
        assert node.id == "custom-id"
        assert node.type == "character"

    def test_metadata(self):
        node: Node[str] = Node(content="test", metadata={"source": "user"})
        assert node.metadata["source"] == "user"

    def test_unique_ids(self):
        ids = {Node().id for _ in range(100)}
        assert len(ids) == 100


class TestEdge:
    def test_defaults(self):
        edge = Edge(source="a", target="b", relation="KNOWS")
        assert edge.weight == 1.0
        assert edge.metadata == {}

    def test_custom(self):
        edge = Edge(source="a", target="b", relation="LIKES", weight=0.5, metadata={"x": 1})
        assert edge.weight == 0.5
        assert edge.metadata["x"] == 1

    def test_frozen(self):
        edge = Edge(source="a", target="b", relation="R")
        with pytest.raises(AttributeError):
            edge.weight = 2.0  # type: ignore[misc]
