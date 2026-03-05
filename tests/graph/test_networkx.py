"""Tests for core NetworkX graph backend."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core.graph._networkx import NetworkXBackend
from ai_arch_toolkit.core.graph._types import Edge, Node


class TestNodeCRUD:
    async def test_add_and_get(self, backend: NetworkXBackend):
        node: Node[str] = Node(id="n1", content="hello")
        await backend.add_node(node)
        got = await backend.get_node("n1")
        assert got is not None
        assert got.content == "hello"

    async def test_get_missing(self, backend: NetworkXBackend):
        assert await backend.get_node("nope") is None

    async def test_update(self, backend: NetworkXBackend):
        await backend.add_node(Node(id="n1", content="old"))
        updated = await backend.update_node("n1", content="new")
        assert updated is not None
        assert updated.content == "new"

    async def test_update_missing(self, backend: NetworkXBackend):
        assert await backend.update_node("nope", content="x") is None

    async def test_remove(self, backend: NetworkXBackend):
        await backend.add_node(Node(id="n1"))
        assert await backend.remove_node("n1") is True
        assert await backend.get_node("n1") is None

    async def test_remove_missing(self, backend: NetworkXBackend):
        assert await backend.remove_node("nope") is False

    async def test_list_and_count(self, backend: NetworkXBackend):
        await backend.add_node(Node(id="a", type="person"))
        await backend.add_node(Node(id="b", type="place"))
        await backend.add_node(Node(id="c", type="person"))
        assert await backend.count_nodes() == 3
        assert await backend.count_nodes(type="person") == 2
        all_nodes = await backend.list_nodes()
        assert len(all_nodes) == 3
        people = await backend.list_nodes(type="person")
        assert len(people) == 2
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

    async def test_remove_edge(self, backend: NetworkXBackend):
        await backend.add_node(Node(id="a"))
        await backend.add_node(Node(id="b"))
        await backend.add_edge(Edge(source="a", target="b", relation="R"))
        assert await backend.remove_edge("a", "b", "R") is True
        assert await backend.remove_edge("a", "b", "R") is False

    async def test_both_directions(self, backend: NetworkXBackend):
        await backend.add_node(Node(id="a"))
        await backend.add_node(Node(id="b"))
        await backend.add_edge(Edge(source="a", target="b", relation="R"))
        both = await backend.get_edges("a", direction="both")
        assert len(both) == 1
        both_b = await backend.get_edges("b", direction="both")
        assert len(both_b) == 1


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

    async def test_neighbors_missing(self, backend: NetworkXBackend):
        assert await backend.neighbors("nope") == []


class TestAlgorithms:
    async def test_bfs(self, backend: NetworkXBackend):
        for i in range(3):
            await backend.add_node(Node(id=f"n{i}"))
        await backend.add_edge(Edge(source="n0", target="n1", relation="R"))
        await backend.add_edge(Edge(source="n1", target="n2", relation="R"))
        bfs = await backend.bfs("n0")
        assert len(bfs) == 3
        assert bfs[0].id == "n0"

    async def test_dfs(self, backend: NetworkXBackend):
        for i in range(3):
            await backend.add_node(Node(id=f"n{i}"))
        await backend.add_edge(Edge(source="n0", target="n1", relation="R"))
        await backend.add_edge(Edge(source="n1", target="n2", relation="R"))
        dfs = await backend.dfs("n0")
        assert len(dfs) == 3

    async def test_bfs_dfs_missing(self, backend: NetworkXBackend):
        assert await backend.bfs("nope") == []
        assert await backend.dfs("nope") == []

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

    async def test_shortest_path_missing(self, backend: NetworkXBackend):
        assert await backend.shortest_path("a", "b") is None

    async def test_centrality(self, backend: NetworkXBackend):
        for i in range(3):
            await backend.add_node(Node(id=f"n{i}"))
        await backend.add_edge(Edge(source="n0", target="n1", relation="R"))
        await backend.add_edge(Edge(source="n0", target="n2", relation="R"))
        c = await backend.centrality()
        assert c["n0"] > 0

    async def test_connected_components(self, backend: NetworkXBackend):
        await backend.add_node(Node(id="a"))
        await backend.add_node(Node(id="b"))
        await backend.add_node(Node(id="c"))
        await backend.add_edge(Edge(source="a", target="b", relation="R"))
        components = await backend.connected_components()
        assert len(components) == 2  # {a,b} and {c}

    async def test_subgraph(self, backend: NetworkXBackend):
        for i in range(3):
            await backend.add_node(Node(id=f"n{i}"))
        await backend.add_edge(Edge(source="n0", target="n1", relation="R"))
        sub = await backend.subgraph(["n0", "n1"])
        assert await sub.count_nodes() == 2

    async def test_relation_filter(self, backend: NetworkXBackend):
        for i in range(3):
            await backend.add_node(Node(id=f"n{i}"))
        await backend.add_edge(Edge(source="n0", target="n1", relation="A"))
        await backend.add_edge(Edge(source="n0", target="n2", relation="B"))
        bfs = await backend.bfs("n0", relation="A")
        assert len(bfs) == 2  # n0 + n1 only


class TestBulk:
    async def test_clear_all(self, backend: NetworkXBackend):
        await backend.add_node(Node(id="a"))
        await backend.add_node(Node(id="b"))
        count = await backend.clear()
        assert count == 2
        assert await backend.count_nodes() == 0

    async def test_clear_by_type(self, backend: NetworkXBackend):
        await backend.add_node(Node(id="a", type="person"))
        await backend.add_node(Node(id="b", type="place"))
        count = await backend.clear(type="person")
        assert count == 1
        assert await backend.count_nodes() == 1
