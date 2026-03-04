"""Tests for memory presets."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.memory._presets import cognitive
from ai_arch_toolkit.toolkit.memory._views import SimilarityView, TemporalView
from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend
from ai_arch_toolkit.toolkit.memory.graph._store import GraphStore
from tests.memory.conftest import make_node


class TestCognitivePreset:
    async def test_views(self):
        store = GraphStore(NetworkXBackend())
        preset = cognitive(store)
        assert "semantic" in preset.views
        assert "episodic" in preset.views
        assert "procedural" in preset.views
        assert "relations" in preset.views
        assert "properties" in preset.views
        assert isinstance(preset["episodic"], TemporalView)
        assert isinstance(preset["semantic"], SimilarityView)

    async def test_getitem(self):
        store = GraphStore(NetworkXBackend())
        preset = cognitive(store)
        view = preset["semantic"]
        assert view is not None


class TestConsolidate:
    async def test_dedup(self):
        store = GraphStore(NetworkXBackend())
        await store.add(make_node(id="a", content={"text": "hello"}))
        await store.add(make_node(id="b", content={"text": "hello"}))
        await store.add(make_node(id="c", content={"text": "world"}))
        preset = cognitive(store)
        removed = await preset.consolidate()
        assert removed == 1
        assert await store.count() == 2
