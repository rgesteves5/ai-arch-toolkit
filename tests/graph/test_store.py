"""Tests for core Graph facade."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from ai_arch_toolkit.core.graph._networkx import NetworkXBackend
from ai_arch_toolkit.core.graph._store import Graph
from ai_arch_toolkit.core.graph._types import Node


class TestNodeOps:
    async def test_add_and_get(self, graph: Graph):
        node: Node[str] = Node(id="n1", content="hello")
        added = await graph.add(node)
        assert added.id == "n1"
        got = await graph.get("n1")
        assert got is not None
        assert got.content == "hello"

    async def test_get_missing(self, graph: Graph):
        assert await graph.get("nope") is None

    async def test_update(self, graph: Graph):
        await graph.add(Node(id="n1", content="old"))
        updated = await graph.update("n1", content="new")
        assert updated is not None
        assert updated.content == "new"

    async def test_update_missing(self, graph: Graph):
        assert await graph.update("nope", content="x") is None

    async def test_remove(self, graph: Graph):
        await graph.add(Node(id="n1"))
        assert await graph.remove("n1") is True
        assert await graph.get("n1") is None

    async def test_remove_missing(self, graph: Graph):
        assert await graph.remove("nope") is False


class TestTypeIndex:
    async def test_fast_lookup_by_type(self, graph: Graph):
        await graph.add(Node(id="a", type="person"))
        await graph.add(Node(id="b", type="place"))
        await graph.add(Node(id="c", type="person"))
        assert await graph.count(type="person") == 2
        people = await graph.list(type="person")
        assert len(people) == 2

    async def test_type_index_updated_on_remove(self, graph: Graph):
        await graph.add(Node(id="a", type="person"))
        await graph.remove("a")
        assert await graph.count(type="person") == 0

    async def test_type_index_updated_on_type_change(self, graph: Graph):
        await graph.add(Node(id="a", type="person"))
        await graph.update("a", type="place")
        assert await graph.count(type="person") == 0
        assert await graph.count(type="place") == 1


class TestEdgeOps:
    async def test_connect_and_edges(self, graph: Graph):
        await graph.add(Node(id="a"))
        await graph.add(Node(id="b"))
        edge = await graph.connect("a", "b", "KNOWS")
        assert edge.relation == "KNOWS"
        edges = await graph.edges("a")
        assert len(edges) == 1

    async def test_disconnect(self, graph: Graph):
        await graph.add(Node(id="a"))
        await graph.add(Node(id="b"))
        await graph.connect("a", "b", "R")
        assert await graph.disconnect("a", "b", "R") is True
        assert await graph.disconnect("a", "b", "R") is False


class TestAlgorithms:
    async def test_bfs_dfs(self, graph: Graph):
        for i in range(3):
            await graph.add(Node(id=f"n{i}"))
        await graph.connect("n0", "n1", "R")
        await graph.connect("n1", "n2", "R")
        bfs = await graph.bfs("n0")
        assert len(bfs) == 3
        dfs = await graph.dfs("n0")
        assert len(dfs) == 3

    async def test_shortest_path(self, graph: Graph):
        for i in range(3):
            await graph.add(Node(id=f"n{i}"))
        await graph.connect("n0", "n1", "R")
        await graph.connect("n1", "n2", "R")
        path = await graph.shortest_path("n0", "n2")
        assert path is not None
        assert len(path) == 3

    async def test_has_algorithms(self, graph: Graph):
        assert graph.has_algorithms

    async def test_centrality(self, graph: Graph):
        await graph.add(Node(id="a"))
        await graph.add(Node(id="b"))
        await graph.connect("a", "b", "R")
        c = await graph.centrality()
        assert isinstance(c, dict)

    async def test_connected_components(self, graph: Graph):
        await graph.add(Node(id="a"))
        await graph.add(Node(id="b"))
        comps = await graph.connected_components()
        assert len(comps) == 2

    async def test_no_algorithms_raises_type_error(self):
        """Algorithm methods raise TypeError when backend doesn't implement them."""
        backend = AsyncMock()
        backend.add_node = AsyncMock()
        backend.get_node = AsyncMock(return_value=None)
        # AsyncMock passes isinstance checks for Protocol by default,
        # so we need a plain object that only has GraphBackend methods.

        class MinimalBackend:
            async def add_node(self, node): ...
            async def get_node(self, node_id): ...
            async def update_node(self, node_id, **attrs): ...
            async def remove_node(self, node_id): ...
            async def list_nodes(self, *, type=None, limit=None): ...
            async def count_nodes(self, *, type=None): ...
            async def add_edge(self, edge): ...
            async def get_edges(self, node_id, *, direction="out", relation=None): ...
            async def remove_edge(self, source, target, relation): ...
            async def neighbors(self, node_id, *, depth=1, relation=None): ...
            async def clear(self, *, type=None): ...

        g = Graph(MinimalBackend())  # type: ignore[arg-type]
        assert not g.has_algorithms
        with pytest.raises(TypeError, match="GraphAlgorithms"):
            await g.bfs("n0")
        with pytest.raises(TypeError, match="GraphAlgorithms"):
            await g.dfs("n0")
        with pytest.raises(TypeError, match="GraphAlgorithms"):
            await g.shortest_path("a", "b")
        with pytest.raises(TypeError, match="GraphAlgorithms"):
            await g.centrality()
        with pytest.raises(TypeError, match="GraphAlgorithms"):
            await g.connected_components()


class TestNewFacadeMethods:
    async def test_has(self, graph: Graph):
        await graph.add(Node(id="n1"))
        assert await graph.has("n1") is True
        assert await graph.has("nope") is False

    async def test_degree(self, graph: Graph):
        await graph.add(Node(id="a"))
        await graph.add(Node(id="b"))
        await graph.connect("a", "b", "R")
        assert await graph.degree("a") == 1
        assert await graph.degree("nope") == 0

    async def test_get_edges_between(self, graph: Graph):
        await graph.add(Node(id="a"))
        await graph.add(Node(id="b"))
        await graph.connect("a", "b", "KNOWS")
        await graph.connect("a", "b", "LIKES")
        edges = await graph.get_edges_between("a", "b")
        assert len(edges) == 2
        edges_filtered = await graph.get_edges_between("a", "b", relation="KNOWS")
        assert len(edges_filtered) == 1

    async def test_list_edges(self, graph: Graph):
        await graph.add(Node(id="a"))
        await graph.add(Node(id="b"))
        await graph.connect("a", "b", "KNOWS")
        await graph.connect("a", "b", "LIKES")
        all_edges = await graph.list_edges()
        assert len(all_edges) == 2
        filtered = await graph.list_edges(relation="KNOWS")
        assert len(filtered) == 1

    async def test_get_orphan_nodes(self, graph: Graph):
        await graph.add(Node(id="a"))
        await graph.add(Node(id="b"))
        await graph.add(Node(id="c"))
        await graph.connect("a", "b", "R")
        orphans = await graph.get_orphan_nodes()
        assert len(orphans) == 1
        assert orphans[0].id == "c"

    async def test_get_stats(self, graph: Graph):
        await graph.add(Node(id="a", type="person"))
        await graph.add(Node(id="b", type="place"))
        await graph.connect("a", "b", "VISITS")
        stats = await graph.get_stats()
        assert stats["node_count"] == 2
        assert stats["edge_count"] == 1
        assert stats["node_types"] == {"person": 1, "place": 1}
        assert stats["edge_relations"] == {"VISITS": 1}

    async def test_filter_nodes(self, graph: Graph):
        await graph.add(Node(id="a", type="person"))
        await graph.add(Node(id="b", type="place"))
        await graph.add(Node(id="c", type="person"))
        people = await graph.filter_nodes(lambda n: n.type == "person")
        assert len(people) == 2

    async def test_filter_edges(self, graph: Graph):
        await graph.add(Node(id="a"))
        await graph.add(Node(id="b"))
        await graph.connect("a", "b", "KNOWS", weight=0.5)
        await graph.connect("a", "b", "LIKES", weight=0.9)
        strong = await graph.filter_edges(lambda e: e.weight > 0.7)
        assert len(strong) == 1
        assert strong[0].relation == "LIKES"

    async def test_copy(self, graph: Graph):
        await graph.add(Node(id="a"))
        await graph.add(Node(id="b"))
        await graph.connect("a", "b", "R")
        clone = await graph.copy()
        assert await clone.count() == 2
        edges = await clone.edges("a")
        assert len(edges) == 1
        # Verify independence
        await graph.remove("a")
        assert await clone.has("a") is True

    async def test_node_count_edge_count_is_empty(self, graph: Graph):
        assert await graph.is_empty() is True
        assert await graph.node_count() == 0
        assert await graph.edge_count() == 0
        await graph.add(Node(id="a"))
        await graph.add(Node(id="b"))
        await graph.connect("a", "b", "R")
        assert await graph.node_count() == 2
        assert await graph.edge_count() == 1
        assert await graph.is_empty() is False


class TestNewAlgorithmFacade:
    async def test_find_all_paths(self, graph: Graph):
        for i in range(4):
            await graph.add(Node(id=f"n{i}"))
        await graph.connect("n0", "n1", "R")
        await graph.connect("n1", "n3", "R")
        await graph.connect("n0", "n2", "R")
        await graph.connect("n2", "n3", "R")
        paths = await graph.find_all_paths("n0", "n3")
        assert len(paths) == 2

    async def test_get_ancestors(self, graph: Graph):
        for i in range(3):
            await graph.add(Node(id=f"n{i}"))
        await graph.connect("n0", "n1", "R")
        await graph.connect("n1", "n2", "R")
        anc = await graph.get_ancestors("n2")
        assert anc == {"n0", "n1"}

    async def test_get_descendants(self, graph: Graph):
        for i in range(3):
            await graph.add(Node(id=f"n{i}"))
        await graph.connect("n0", "n1", "R")
        await graph.connect("n1", "n2", "R")
        desc = await graph.get_descendants("n0")
        assert desc == {"n1", "n2"}

    async def test_get_subgraph(self, graph: Graph):
        for i in range(3):
            await graph.add(Node(id=f"n{i}"))
        await graph.connect("n0", "n1", "R")
        sub = await graph.get_subgraph(["n0", "n1"])
        assert await sub.count() == 2
        assert isinstance(sub, Graph)

    async def test_get_ego_graph(self, graph: Graph):
        for i in range(4):
            await graph.add(Node(id=f"n{i}"))
        await graph.connect("n0", "n1", "R")
        await graph.connect("n1", "n2", "R")
        await graph.connect("n2", "n3", "R")
        ego = await graph.get_ego_graph("n0", radius=1)
        assert isinstance(ego, Graph)
        assert await ego.count() == 2  # n0 + n1

    async def test_pagerank(self, graph: Graph):
        for i in range(3):
            await graph.add(Node(id=f"n{i}"))
        await graph.connect("n0", "n1", "R")
        await graph.connect("n1", "n2", "R")
        pr = await graph.pagerank()
        assert len(pr) == 3
        assert all(v > 0 for v in pr.values())

    async def test_algorithms_raise_without_support(self):
        class MinimalBackend:
            async def add_node(self, node): ...
            async def get_node(self, node_id): ...
            async def update_node(self, node_id, **attrs): ...
            async def remove_node(self, node_id): ...
            async def list_nodes(self, *, type=None, limit=None): ...
            async def count_nodes(self, *, type=None): ...
            async def add_edge(self, edge): ...
            async def get_edges(self, node_id, *, direction="out", relation=None): ...
            async def remove_edge(self, source, target, relation): ...
            async def neighbors(self, node_id, *, depth=1, relation=None): ...
            async def clear(self, *, type=None): ...

        g = Graph(MinimalBackend())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="GraphAlgorithms"):
            await g.find_all_paths("a", "b")
        with pytest.raises(TypeError, match="GraphAlgorithms"):
            await g.get_ancestors("a")
        with pytest.raises(TypeError, match="GraphAlgorithms"):
            await g.get_descendants("a")
        with pytest.raises(TypeError, match="GraphAlgorithms"):
            await g.get_subgraph(["a"])
        with pytest.raises(TypeError, match="GraphAlgorithms"):
            await g.get_ego_graph("a")
        with pytest.raises(TypeError, match="GraphAlgorithms"):
            await g.pagerank()


class TestPersistence:
    async def test_to_dict_from_dict(self, graph: Graph):
        await graph.add(Node(id="n1", type="person", content={"name": "Alice"}))
        await graph.add(Node(id="n2", type="place", content="Wonderland"))
        await graph.connect("n1", "n2", "VISITS")
        data = await graph.to_dict()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        restored = await Graph.from_dict(data, NetworkXBackend())
        assert await restored.count() == 2
        edges = await restored.edges("n1")
        assert len(edges) == 1

    async def test_save_load(self, graph: Graph, tmp_path):
        await graph.add(Node(id="n1", content="persist"))
        path = tmp_path / "graph.json"
        await graph.save(path)
        loaded = await Graph.load(path, NetworkXBackend())
        node = await loaded.get("n1")
        assert node is not None
        assert node.content == "persist"

    async def test_content_loader(self):
        @dataclass(frozen=True, slots=True)
        class Character:
            name: str

        g = Graph(NetworkXBackend())
        await g.add(Node(id="n1", type="char", content=Character(name="Alice")))
        data = await g.to_dict()
        assert data["nodes"][0]["content"] == {"name": "Alice"}
        restored = await Graph.from_dict(
            data,
            NetworkXBackend(),
            content_loader=lambda d: Character(**d) if isinstance(d, dict) else d,
        )
        node = await restored.get("n1")
        assert node is not None
        assert isinstance(node.content, Character)
        assert node.content.name == "Alice"

    async def test_edge_weight_metadata_preserved(self, graph: Graph):
        await graph.add(Node(id="a"))
        await graph.add(Node(id="b"))
        await graph.connect("a", "b", "LIKES", weight=0.75, metadata={"reason": "fun"})
        data = await graph.to_dict()
        restored = await Graph.from_dict(data, NetworkXBackend())
        edges = await restored.edges("a")
        assert edges[0].weight == 0.75
        assert edges[0].metadata["reason"] == "fun"


class TestBulk:
    async def test_clear(self, graph: Graph):
        await graph.add(Node(id="a"))
        await graph.add(Node(id="b"))
        count = await graph.clear()
        assert count == 2
        assert await graph.count() == 0

    async def test_clear_by_type(self, graph: Graph):
        await graph.add(Node(id="a", type="person"))
        await graph.add(Node(id="b", type="place"))
        count = await graph.clear(type="person")
        assert count == 1
        assert await graph.count() == 1

    async def test_add_many(self, graph: Graph):
        nodes = [Node(type="person", content=f"node {i}") for i in range(5)]
        count = await graph.add_many(nodes)
        assert count == 5
        assert await graph.count() == 5

    async def test_remove_many(self, graph: Graph):
        nodes = [Node(id=f"n{i}") for i in range(3)]
        await graph.add_many(nodes)
        count = await graph.remove_many(["n0", "n1", "n99"])
        assert count == 2


class TestSyncWrappers:
    def test_add_get_sync(self):
        g = Graph(NetworkXBackend())
        node = g.add_sync(Node(id="n1", content="hello"))
        assert node.id == "n1"
        got = g.get_sync("n1")
        assert got is not None
        assert got.content == "hello"

    def test_connect_sync(self):
        g = Graph(NetworkXBackend())
        g.add_sync(Node(id="a"))
        g.add_sync(Node(id="b"))
        edge = g.connect_sync("a", "b", "R")
        assert edge.relation == "R"

    def test_neighbors_sync(self):
        g = Graph(NetworkXBackend())
        g.add_sync(Node(id="a"))
        g.add_sync(Node(id="b"))
        g.connect_sync("a", "b", "R")
        neighbors = g.neighbors_sync("a")
        assert len(neighbors) == 1

    def test_save_load_sync(self, tmp_path):
        g = Graph(NetworkXBackend())
        g.add_sync(Node(id="n1", content="test"))
        path = tmp_path / "graph.json"
        g.save_sync(path)
        loaded = Graph.load_sync(path, NetworkXBackend())
        node = loaded.get_sync("n1")
        assert node is not None
        assert node.content == "test"

    def test_from_dict_sync(self):
        g = Graph(NetworkXBackend())
        g.add_sync(Node(id="n1", content="hello"))
        g.add_sync(Node(id="n2", content="world"))
        g.connect_sync("n1", "n2", "R")
        data = _run_sync(g.to_dict())
        restored = Graph.from_dict_sync(data, NetworkXBackend())
        assert restored.count_sync() == 2


def _run_sync(coro):
    """Helper to run async in sync test context."""
    from ai_arch_toolkit.core._sync import _run_sync as rs

    return rs(coro)
