"""Tests for view classes and composite_score."""

from __future__ import annotations

from datetime import timedelta

from ai_arch_toolkit.toolkit.memory._types import Node, SearchResult, _now_utc
from ai_arch_toolkit.toolkit.memory._views import (
    PropertyView,
    RelationalView,
    SimilarityView,
    TemporalView,
    composite_score,
)
from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend
from ai_arch_toolkit.toolkit.memory.graph._store import GraphStore
from tests.memory.conftest import make_node, mock_embed_fn


class TestTemporalView:
    async def test_recent(self):
        store = GraphStore(NetworkXBackend())
        now = _now_utc()
        for i in range(5):
            node = Node(
                id=f"n{i}",
                type="event",
                content={"text": f"event {i}"},
                timestamp=now - timedelta(hours=i),
            )
            await store.add(node)
        view = TemporalView(store, node_type="event")
        recent = await view.recent(k=3)
        assert len(recent) == 3
        assert recent[0].id == "n0"  # most recent

    async def test_since(self):
        store = GraphStore(NetworkXBackend())
        now = _now_utc()
        await store.add(Node(id="new", type="event", timestamp=now))
        await store.add(Node(id="old", type="event", timestamp=now - timedelta(hours=48)))
        view = TemporalView(store, node_type="event")
        result = await view.since(hours=24)
        assert len(result) == 1
        assert result[0].id == "new"

    async def test_between(self):
        store = GraphStore(NetworkXBackend())
        now = _now_utc()
        await store.add(Node(id="a", timestamp=now - timedelta(hours=5)))
        await store.add(Node(id="b", timestamp=now - timedelta(hours=2)))
        await store.add(Node(id="c", timestamp=now))
        view = TemporalView(store)
        result = await view.between(now - timedelta(hours=3), now - timedelta(hours=1))
        assert len(result) == 1
        assert result[0].id == "b"

    async def test_append(self):
        store = GraphStore(NetworkXBackend())
        view = TemporalView(store, node_type="event")
        n1 = await view.append({"text": "first"}, source="test")
        await view.append({"text": "second"}, source="test")
        assert n1.type == "event"
        # Check NEXT edge was created
        edges = await store.edges(n1.id, direction="out")
        assert any(e.relation == "NEXT" for e in edges)

    async def test_append_no_link(self):
        store = GraphStore(NetworkXBackend())
        view = TemporalView(store)
        n1 = await view.append({"text": "only"}, link_previous=False)
        edges = await store.edges(n1.id, direction="out")
        assert len(edges) == 0


class TestSimilarityView:
    async def test_find(self):
        embed = mock_embed_fn()
        store = GraphStore(NetworkXBackend(), embed=embed)
        await store.add(make_node(id="n1", type="fact", content={"text": "python lang"}))
        await store.add(make_node(id="n2", type="fact", content={"text": "java lang"}))
        view = SimilarityView(store, node_type="fact")
        results = await view.find("python")
        assert len(results) >= 1

    async def test_similar_to(self):
        embed = mock_embed_fn()
        store = GraphStore(NetworkXBackend(), embed=embed)
        await store.add(make_node(id="n1", type="fact", content={"text": "python programming"}))
        await store.add(make_node(id="n2", type="fact", content={"text": "java coding"}))
        view = SimilarityView(store, node_type="fact")
        results = await view.similar_to("n1")
        # Should not include n1 itself
        assert all(r.node.id != "n1" for r in results)


class TestRelationalView:
    async def test_neighbors(self):
        store = GraphStore(NetworkXBackend())
        await store.add(make_node(id="a"))
        await store.add(make_node(id="b"))
        await store.connect("a", "b", "KNOWS")
        view = RelationalView(store)
        neighbors = await view.neighbors("a")
        assert len(neighbors) == 1

    async def test_path(self):
        store = GraphStore(NetworkXBackend())
        for i in range(3):
            await store.add(make_node(id=f"n{i}"))
        await store.connect("n0", "n1", "R")
        await store.connect("n1", "n2", "R")
        view = RelationalView(store)
        path = await view.path("n0", "n2")
        assert path is not None
        assert len(path) == 3

    async def test_connect_disconnect(self):
        store = GraphStore(NetworkXBackend())
        await store.add(make_node(id="a"))
        await store.add(make_node(id="b"))
        view = RelationalView(store)
        await view.connect("a", "b", "LIKES")
        edges = await view.edges("a")
        assert len(edges) == 1
        await view.disconnect("a", "b", "LIKES")
        edges = await view.edges("a")
        assert len(edges) == 0


class TestPropertyView:
    async def test_filter_by_metadata(self):
        store = GraphStore(NetworkXBackend())
        await store.add(make_node(id="a", metadata={"topic": "ml"}))
        await store.add(make_node(id="b", metadata={"topic": "web"}))
        view = PropertyView(store)
        results = await view.filter(topic="ml")
        assert len(results) == 1
        assert results[0].id == "a"

    async def test_by_confidence(self):
        store = GraphStore(NetworkXBackend())
        await store.add(make_node(id="high", confidence=0.9))
        await store.add(make_node(id="low", confidence=0.2))
        view = PropertyView(store)
        results = await view.by_confidence(min_confidence=0.5)
        assert len(results) == 1
        assert results[0].id == "high"

    async def test_by_source(self):
        store = GraphStore(NetworkXBackend())
        await store.add(make_node(id="a", source="user_stated"))
        await store.add(make_node(id="b", source="agent_inferred"))
        view = PropertyView(store)
        results = await view.by_source("user_stated")
        assert len(results) == 1


class TestCompositeScore:
    def test_returns_weighted_sum(self):
        node = Node(
            content={"text": "test"},
            timestamp=_now_utc(),
            access_count=10,
        )
        result = SearchResult(node=node, score=0.9)
        score = composite_score(result)
        assert 0 < score <= 1.0

    def test_older_nodes_score_lower(self):
        now = _now_utc()
        recent = SearchResult(
            node=Node(content={"text": "new"}, timestamp=now, access_count=0),
            score=0.8,
        )
        old = SearchResult(
            node=Node(
                content={"text": "old"},
                timestamp=now - timedelta(days=30),
                access_count=0,
            ),
            score=0.8,
        )
        assert composite_score(recent) > composite_score(old)
