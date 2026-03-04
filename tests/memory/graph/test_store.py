"""Tests for GraphStore facade."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend
from ai_arch_toolkit.toolkit.memory.graph._store import GraphStore
from tests.memory.conftest import make_node, mock_embed_fn


class TestAutoEmbed:
    async def test_auto_embeds_on_add(self):
        embed = mock_embed_fn()
        store = GraphStore(NetworkXBackend(), embed=embed)
        node = make_node(content={"text": "hello world"})
        added = await store.add(node)
        embed.assert_called_once()
        assert added.embedding is not None

    async def test_no_embed_without_fn(self):
        store = GraphStore(NetworkXBackend())
        node = make_node(content={"text": "hello"})
        added = await store.add(node)
        assert added.embedding is None

    async def test_re_embeds_on_content_update(self):
        embed = mock_embed_fn()
        store = GraphStore(NetworkXBackend(), embed=embed)
        node = make_node(id="n1", content={"text": "old"})
        await store.add(node)
        await store.update("n1", content={"text": "new"})
        assert embed.call_count == 2  # once for add, once for update


class TestSearchCascade:
    async def test_keyword_fallback(self):
        store = GraphStore(NetworkXBackend())
        await store.add(make_node(id="n1", content={"text": "python programming"}))
        await store.add(make_node(id="n2", content={"text": "java coding"}))
        results = await store.search("python")
        assert len(results) >= 1
        assert results[0].node.id == "n1"

    async def test_index_search(self):
        embed = mock_embed_fn()
        store = GraphStore(NetworkXBackend(), embed=embed)
        await store.add(make_node(id="n1", content={"text": "hello world"}))
        await store.add(make_node(id="n2", content={"text": "goodbye moon"}))
        results = await store.search("hello world")
        assert len(results) >= 1

    async def test_native_vector_precedence(self):
        """If backend.search_similar returns results, index is not used."""
        backend = AsyncMock()
        backend.add_node = AsyncMock()
        backend.search_similar = AsyncMock(
            return_value=[make_node(id="native", content={"text": "native"})]
        )
        embed = mock_embed_fn()
        store = GraphStore(backend, embed=embed)
        results = await store.search("test")
        assert len(results) == 1
        assert results[0].node.id == "native"


class TestAccessTracking:
    async def test_get_bumps_access(self):
        store = GraphStore(NetworkXBackend())
        await store.add(make_node(id="n1"))
        node = await store.get("n1")
        assert node is not None
        assert node.access_count == 1
        assert node.last_accessed is not None
        node2 = await store.get("n1")
        assert node2 is not None
        assert node2.access_count == 2


class TestTypeIndex:
    async def test_fast_lookup_by_type(self):
        store = GraphStore(NetworkXBackend())
        await store.add(make_node(id="a", type="fact"))
        await store.add(make_node(id="b", type="event"))
        await store.add(make_node(id="c", type="fact"))
        assert await store.count(type="fact") == 2
        facts = await store.list(type="fact")
        assert len(facts) == 2

    async def test_type_index_updated_on_remove(self):
        store = GraphStore(NetworkXBackend())
        await store.add(make_node(id="a", type="fact"))
        await store.remove("a")
        assert await store.count(type="fact") == 0


class TestPersistence:
    async def test_to_dict_from_dict(self):
        store = GraphStore(NetworkXBackend())
        await store.add(make_node(id="n1", type="fact", content={"text": "hello"}))
        await store.add(make_node(id="n2", type="event", content={"text": "world"}))
        await store.connect("n1", "n2", "RELATED")
        data = await store.to_dict()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        # Restore
        new_store = await GraphStore.from_dict(data, NetworkXBackend())
        assert await new_store.count() == 2
        edges = await new_store.edges("n1")
        assert len(edges) == 1

    async def test_save_load(self, tmp_path):
        store = GraphStore(NetworkXBackend())
        await store.add(make_node(id="n1", content={"text": "persist"}))
        path = tmp_path / "graph.json"
        await store.save(path)
        loaded = await GraphStore.load(path, NetworkXBackend())
        node = await loaded.backend.get_node("n1")
        assert node is not None
        assert node.content["text"] == "persist"


class TestBulk:
    async def test_clear_by_type(self):
        store = GraphStore(NetworkXBackend())
        await store.add(make_node(id="a", type="fact"))
        await store.add(make_node(id="b", type="event"))
        count = await store.clear(type="fact")
        assert count == 1
        assert await store.count() == 1

    async def test_index_synced_on_add_remove(self):
        embed = mock_embed_fn()
        store = GraphStore(NetworkXBackend(), embed=embed)
        await store.add(make_node(id="n1", content={"text": "hello"}))
        assert await store._index.count() == 1
        await store.remove("n1")
        assert await store._index.count() == 0

    async def test_add_many(self):
        store = GraphStore(NetworkXBackend())
        nodes = [make_node(type="fact", content={"text": f"node {i}"}) for i in range(5)]
        count = await store.add_many(nodes)
        assert count == 5
        assert await store.count() == 5

    async def test_remove_many(self):
        store = GraphStore(NetworkXBackend())
        nodes = [make_node(id=f"n{i}") for i in range(3)]
        await store.add_many(nodes)
        count = await store.remove_many(["n0", "n1", "n99"])
        assert count == 2


class TestClearWithEmbeddings:
    async def test_clear_all_removes_embeddings(self):
        embed = mock_embed_fn()
        store = GraphStore(NetworkXBackend(), embed=embed)
        await store.add(make_node(id="a", content={"text": "hello"}))
        await store.add(make_node(id="b", content={"text": "world"}))
        assert await store._index.count() == 2
        await store.clear()
        assert await store._index.count() == 0

    async def test_clear_by_type_removes_embeddings(self):
        embed = mock_embed_fn()
        store = GraphStore(NetworkXBackend(), embed=embed)
        await store.add(make_node(id="a", type="fact", content={"text": "hello"}))
        await store.add(make_node(id="b", type="event", content={"text": "world"}))
        assert await store._index.count() == 2
        await store.clear(type="fact")
        assert await store._index.count() == 1

    async def test_clear_preserves_custom_index(self):
        """clear() must not replace a custom VectorIndex with BruteForceIndex."""
        from ai_arch_toolkit.toolkit.memory.graph._index import BruteForceIndex

        embed = mock_embed_fn()
        custom_index = BruteForceIndex()  # stand-in for a custom impl
        store = GraphStore(NetworkXBackend(), embed=embed, index=custom_index)
        await store.add(make_node(id="a", content={"text": "hello"}))
        await store.clear()
        # The index instance must be the same object, not replaced
        assert store._index is custom_index


class TestPersistenceRoundtrip:
    async def test_edge_weight_metadata_preserved(self):
        store = GraphStore(NetworkXBackend())
        await store.add(make_node(id="a"))
        await store.add(make_node(id="b"))
        await store.connect("a", "b", "LIKES", weight=0.75, metadata={"reason": "fun"})
        data = await store.to_dict()
        restored = await GraphStore.from_dict(data, NetworkXBackend())
        edges = await restored.edges("a")
        assert len(edges) == 1
        assert edges[0].weight == 0.75
        assert edges[0].metadata["reason"] == "fun"

    async def test_node_lifecycle_fields_preserved(self):
        store = GraphStore(NetworkXBackend())
        node = make_node(
            id="n1",
            type="fact",
            content={"text": "test"},
            source="user_stated",
            confidence=0.8,
        )
        await store.add(node)
        data = await store.to_dict()
        restored = await GraphStore.from_dict(data, NetworkXBackend())
        got = await restored.backend.get_node("n1")
        assert got is not None
        assert got.source == "user_stated"
        assert got.confidence == 0.8
        assert got.type == "fact"


class TestProperties:
    async def test_has_embeddings(self):
        assert not GraphStore(NetworkXBackend()).has_embeddings
        assert GraphStore(NetworkXBackend(), embed=mock_embed_fn()).has_embeddings

    async def test_has_algorithms(self):
        assert GraphStore(NetworkXBackend()).has_algorithms
