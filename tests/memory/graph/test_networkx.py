"""Tests for NetworkX graph backend."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.toolkit.memory._types import Edge, Node
from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend


@pytest.fixture
def backend() -> NetworkXBackend:
    return NetworkXBackend()


class TestNodeCRUD:
    async def test_add_and_get(self, backend: NetworkXBackend):
        node = Node(id="n1", content={"text": "hello"})
        await backend.add_node(node)
        got = await backend.get_node("n1")
        assert got is not None
        assert got.content["text"] == "hello"

    async def test_get_missing(self, backend: NetworkXBackend):
        assert await backend.get_node("nope") is None

    async def test_update(self, backend: NetworkXBackend):
        await backend.add_node(Node(id="n1", content={"text": "old"}))
        updated = await backend.update_node("n1", content={"text": "new"})
        assert updated is not None
        assert updated.content["text"] == "new"

    async def test_update_missing(self, backend: NetworkXBackend):
        assert await backend.update_node("nope", content={}) is None

    async def test_remove(self, backend: NetworkXBackend):
        await backend.add_node(Node(id="n1"))
        assert await backend.remove_node("n1") is True
        assert await backend.get_node("n1") is None

    async def test_remove_missing(self, backend: NetworkXBackend):
        assert await backend.remove_node("nope") is False

    async def test_list_and_count(self, backend: NetworkXBackend):
        await backend.add_node(Node(id="a", type="fact"))
        await backend.add_node(Node(id="b", type="event"))
        await backend.add_node(Node(id="c", type="fact"))
        assert await backend.count_nodes() == 3
        assert await backend.count_nodes(type="fact") == 2
        all_nodes = await backend.list_nodes()
        assert len(all_nodes) == 3
        facts = await backend.list_nodes(type="fact")
        assert len(facts) == 2
        limited = await backend.list_nodes(limit=1)
        assert len(limited) == 1


class TestEdgeCRUD:
    async def test_add_and_get_edges(self, backend: NetworkXBackend):
        await backend.add_node(Node(id="a"))
        await backend.add_node(Node(id="b"))
        edge = Edge(source="a", target="b", relation="KNOWS")
        await backend.add_edge(edge)
        out = await backend.get_edges("a", direction="out")
        assert len(out) == 1
        assert out[0].relation == "KNOWS"
        in_edges = await backend.get_edges("b", direction="in")
        assert len(in_edges) == 1

    async def test_add_edge_missing_nodes(self, backend: NetworkXBackend):
        await backend.add_node(Node(id="a"))
        with pytest.raises(ValueError, match="not found"):
            await backend.add_edge(Edge(source="a", target="missing", relation="R"))
        with pytest.raises(ValueError, match="not found"):
            await backend.add_edge(Edge(source="missing", target="a", relation="R"))

    async def test_remove_edge(self, backend: NetworkXBackend):
        await backend.add_node(Node(id="a"))
        await backend.add_node(Node(id="b"))
        await backend.add_edge(Edge(source="a", target="b", relation="R"))
        assert await backend.remove_edge("a", "b", "R") is True
        assert await backend.remove_edge("a", "b", "R") is False


class TestSearch:
    async def test_keyword_search(self, backend: NetworkXBackend):
        await backend.add_node(Node(id="n1", content={"text": "python programming"}))
        await backend.add_node(Node(id="n2", content={"text": "java coding"}))
        results = await backend.search_content("python")
        assert len(results) >= 1
        assert results[0].id == "n1"

    async def test_search_similar_returns_none(self, backend: NetworkXBackend):
        result = await backend.search_similar([1.0, 0.0])
        assert result is None


class TestTraversal:
    async def test_neighbors(self, backend: NetworkXBackend):
        for i in range(4):
            await backend.add_node(Node(id=f"n{i}"))
        await backend.add_edge(Edge(source="n0", target="n1", relation="R"))
        await backend.add_edge(Edge(source="n1", target="n2", relation="R"))
        await backend.add_edge(Edge(source="n0", target="n3", relation="OTHER"))
        neighbors = await backend.neighbors("n0", depth=1)
        assert len(neighbors) == 2
        neighbors_r = await backend.neighbors("n0", depth=1, relation="R")
        assert len(neighbors_r) == 1
        deep = await backend.neighbors("n0", depth=2, relation="R")
        assert len(deep) == 2


class TestAlgorithms:
    async def test_bfs_dfs(self, backend: NetworkXBackend):
        for i in range(3):
            await backend.add_node(Node(id=f"n{i}"))
        await backend.add_edge(Edge(source="n0", target="n1", relation="R"))
        await backend.add_edge(Edge(source="n1", target="n2", relation="R"))
        bfs = await backend.bfs("n0")
        assert len(bfs) == 3
        dfs = await backend.dfs("n0")
        assert len(dfs) == 3

    async def test_shortest_path(self, backend: NetworkXBackend):
        for i in range(3):
            await backend.add_node(Node(id=f"n{i}"))
        await backend.add_edge(Edge(source="n0", target="n1", relation="R"))
        await backend.add_edge(Edge(source="n1", target="n2", relation="R"))
        path = await backend.shortest_path("n0", "n2")
        assert path is not None
        assert len(path) == 3
        no_path = await backend.shortest_path("n2", "n0")
        assert no_path is None


class TestBulk:
    async def test_clear_all(self, backend: NetworkXBackend):
        await backend.add_node(Node(id="a"))
        await backend.add_node(Node(id="b"))
        count = await backend.clear()
        assert count == 2
        assert await backend.count_nodes() == 0

    async def test_clear_by_type(self, backend: NetworkXBackend):
        await backend.add_node(Node(id="a", type="fact"))
        await backend.add_node(Node(id="b", type="event"))
        count = await backend.clear(type="fact")
        assert count == 1
        assert await backend.count_nodes() == 1
