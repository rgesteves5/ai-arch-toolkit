"""Memory tools — ToolGroup for agent-driven memory operations."""

from __future__ import annotations

from ai_arch_toolkit.core._tools import ToolGroup
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.toolkit.memory._types import Node
from ai_arch_toolkit.toolkit.memory.graph._store import GraphStore


def memory_tools(store: GraphStore) -> ToolGroup:
    """Create a ToolGroup with memory CRUD tools bound to a GraphStore.

    Tools:
        - remember: Store a new memory node
        - recall: Search for relevant memories
        - explore_memory: Get a node and its neighbors
        - forget_memory: Remove a memory node
    """

    @tool
    async def remember(
        text: str,
        node_type: str = "fact",
        source: str = "agent_inferred",
        subject: str = "",
    ) -> str:
        """Store a new memory in the graph.

        Args:
            text: The text content to remember.
            node_type: Type of memory node (fact, event, preference, rule, etc.).
            source: How this memory was obtained (user_stated, agent_inferred, tool_result).
            subject: Optional subject or entity this memory is about.
        """
        try:
            content: dict[str, str] = {"text": text}
            if subject:
                content["subject"] = subject
            node = Node(type=node_type, content=content, source=source)
            added = await store.add(node)
            return f"Remembered [{added.type}] id={added.id}: {text[:80]}"
        except Exception as e:
            return f"Failed to remember: {e}"

    @tool
    async def recall(query: str, k: int = 5, node_type: str = "") -> str:
        """Search for relevant memories.

        Args:
            query: Search text to find related memories.
            k: Maximum number of results to return.
            node_type: Optional filter by node type (empty string for all types).
        """
        try:
            type_filter = node_type if node_type else None
            results = await store.search(query, type=type_filter, k=k)
            if not results:
                return "No matching memories found."
            lines: list[str] = []
            for r in results:
                text_parts = [str(v) for v in r.node.content.values() if isinstance(v, str)]
                text = " ".join(text_parts)
                lines.append(f"[{r.node.type}] id={r.node.id} score={r.score:.2f}: {text[:100]}")
            return "\n".join(lines)
        except Exception as e:
            return f"Recall failed: {e}"

    @tool
    async def explore_memory(node_id: str, depth: int = 1) -> str:
        """Explore a memory node and its neighbors in the graph.

        Args:
            node_id: The ID of the node to explore.
            depth: How many hops to traverse from the node.
        """
        try:
            node = await store.get(node_id)
            if node is None:
                return f"Node not found: {node_id}"
            text_parts = [str(v) for v in node.content.values() if isinstance(v, str)]
            text = " ".join(text_parts)
            lines = [f"Node [{node.type}] id={node.id}: {text[:100]}"]
            lines.append(f"  source={node.source}, confidence={node.confidence}")
            lines.append(f"  accessed={node.access_count}x")
            neighbors = await store.neighbors(node_id, depth=depth)
            if neighbors:
                lines.append(f"Neighbors ({len(neighbors)}):")
                for n in neighbors:
                    ntext = " ".join(str(v) for v in n.content.values() if isinstance(v, str))
                    lines.append(f"  [{n.type}] id={n.id}: {ntext[:80]}")
            return "\n".join(lines)
        except Exception as e:
            return f"Explore failed: {e}"

    @tool
    async def forget_memory(node_id: str) -> str:
        """Remove a memory node from the graph.

        Args:
            node_id: The ID of the node to remove.
        """
        try:
            removed = await store.remove(node_id)
            if removed:
                return f"Removed node {node_id}"
            return f"Node not found: {node_id}"
        except Exception as e:
            return f"Forget failed: {e}"

    return ToolGroup(remember, recall, explore_memory, forget_memory)
