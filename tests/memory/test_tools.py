"""Tests for memory tools."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.memory._tools import memory_tools
from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend
from ai_arch_toolkit.toolkit.memory.graph._store import GraphStore
from tests.memory.conftest import make_node


class TestMemoryTools:
    async def test_remember_and_recall(self):
        store = GraphStore(NetworkXBackend())
        tools = memory_tools(store)
        # Remember
        result = await tools.async_execute(
            _make_tool_call("remember", {"text": "Python is great", "node_type": "fact"})
        )
        assert "Remembered" in result
        # Recall
        result = await tools.async_execute(_make_tool_call("recall", {"query": "Python"}))
        assert "Python is great" in result

    async def test_recall_no_results(self):
        store = GraphStore(NetworkXBackend())
        tools = memory_tools(store)
        result = await tools.async_execute(_make_tool_call("recall", {"query": "nonexistent"}))
        assert "No matching memories" in result

    async def test_explore(self):
        store = GraphStore(NetworkXBackend())
        await store.add(make_node(id="n1", content={"text": "main node"}))
        await store.add(make_node(id="n2", content={"text": "neighbor"}))
        await store.connect("n1", "n2", "RELATED")
        tools = memory_tools(store)
        result = await tools.async_execute(_make_tool_call("explore_memory", {"node_id": "n1"}))
        assert "main node" in result
        assert "neighbor" in result

    async def test_forget(self):
        store = GraphStore(NetworkXBackend())
        await store.add(make_node(id="n1", content={"text": "forget me"}))
        tools = memory_tools(store)
        result = await tools.async_execute(_make_tool_call("forget_memory", {"node_id": "n1"}))
        assert "Removed" in result
        assert await store.backend.get_node("n1") is None


def _make_tool_call(name: str, input: dict) -> object:
    """Create a minimal ToolCall-like object."""
    from ai_arch_toolkit.core._response import ToolCall

    return ToolCall(id=f"call_{name}", name=name, input=input)
