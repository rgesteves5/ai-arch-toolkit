"""Private memory helpers for the configurable agent nano project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_arch_toolkit.core._sync import _run_sync
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.memory._types import Node
from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend
from ai_arch_toolkit.toolkit.memory.graph._store import GraphStore


def create_private_memory() -> GraphStore:
    """Create an in-memory private memory store."""
    return GraphStore(NetworkXBackend())


def load_private_memory_sync(path: str | Path) -> GraphStore:
    """Load private memory from disk, or create a new store if missing."""
    memory_path = Path(path)
    if not memory_path.exists():
        return create_private_memory()
    return _run_sync(GraphStore.load(memory_path, NetworkXBackend()))


def save_private_memory_sync(store: GraphStore, path: str | Path) -> None:
    """Save private memory to disk."""
    _run_sync(store.save(Path(path)))


def private_memory_tools(
    store: GraphStore,
    *,
    read: bool = True,
    write: bool = True,
) -> ToolGroup:
    """Return private memory tools with idempotent remember semantics."""

    @tool
    async def remember(
        text: str,
        node_type: str = "fact",
        source: str = "agent_inferred",
        subject: str = "",
    ) -> str:
        """Store a new memory unless an equivalent memory already exists.

        Args:
            text: The text content to remember.
            node_type: Type of memory node (fact, event, preference, rule, etc.).
            source: How this memory was obtained (user_stated, agent_inferred, tool_result).
            subject: Optional subject or entity this memory is about.
        """
        try:
            existing = await _find_equivalent_memory(store, text, node_type, subject)
            if existing is not None:
                return f"Already remembered [{existing.type}] id={existing.id}: {text[:80]}"

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
                result_text = " ".join(text_parts)
                lines.append(
                    f"[{r.node.type}] id={r.node.id} score={r.score:.2f}: {result_text[:100]}"
                )
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

    @tool
    async def list_memories(node_type: str = "", limit: int = 50) -> str:
        """List stored private memories.

        Args:
            node_type: Optional filter by node type (empty string for all types).
            limit: Maximum number of memories to list.
        """
        try:
            type_filter = node_type if node_type else None
            nodes = await store.list(type=type_filter, limit=limit)
            if not nodes:
                return "No memories found."
            lines: list[str] = []
            for node in nodes:
                lines.append(_format_memory_node(node))
            return "\n".join(lines)
        except Exception as e:
            return f"List memories failed: {e}"

    @tool
    async def find_duplicate_memories(node_type: str = "") -> str:
        """Find exact duplicate private memories.

        Args:
            node_type: Optional filter by node type (empty string for all types).
        """
        try:
            groups = await _duplicate_memory_groups(store, node_type or None)
            if not groups:
                return "No duplicate memories found."
            lines = [f"Found {len(groups)} duplicate memory group(s):"]
            for index, nodes in enumerate(groups, start=1):
                lines.append(f"Group {index}:")
                for node in nodes:
                    lines.append(f"  {_format_memory_node(node)}")
            return "\n".join(lines)
        except Exception as e:
            return f"Find duplicates failed: {e}"

    @tool
    async def consolidate_memories(node_type: str = "") -> str:
        """Remove exact duplicate private memories, keeping the oldest node.

        Args:
            node_type: Optional filter by node type (empty string for all types).
        """
        try:
            groups = await _duplicate_memory_groups(store, node_type or None)
            removed = 0
            kept: list[str] = []
            for nodes in groups:
                ordered = sorted(nodes, key=lambda node: node.created_at)
                keep = ordered[0]
                kept.append(str(keep.id))
                for duplicate in ordered[1:]:
                    if await store.remove(duplicate.id):
                        removed += 1
            if removed == 0:
                return "No duplicate memories found."
            return f"Removed {removed} duplicate memory node(s). Kept ids: {', '.join(kept)}"
        except Exception as e:
            return f"Consolidate memories failed: {e}"

    tools = []
    if write:
        tools.append(remember)
    if read:
        tools.extend([recall, explore_memory, list_memories, find_duplicate_memories])
    if write:
        tools.extend([forget_memory, consolidate_memories])
    return ToolGroup(*tools)


async def memory_count(store: GraphStore | None) -> int:
    """Return memory node count."""
    if store is None:
        return 0
    return await store.count()


async def _find_equivalent_memory(
    store: GraphStore,
    text: str,
    node_type: str,
    subject: str,
) -> Node | None:
    normalized_text = _normalize_memory_text(text)
    normalized_subject = _normalize_memory_text(subject)
    nodes = await store.list(type=node_type)

    for node in nodes:
        node_text = str(node.content.get("text", ""))
        node_subject = str(node.content.get("subject", ""))
        if _normalize_memory_text(node_text) != normalized_text:
            continue
        if normalized_subject and _normalize_memory_text(node_subject) != normalized_subject:
            continue
        return node
    return None


def _normalize_memory_text(value: str) -> str:
    return " ".join(value.casefold().strip().rstrip(".").split())


async def _duplicate_memory_groups(
    store: GraphStore,
    node_type: str | None,
) -> list[list[Node]]:
    nodes = await store.list(type=node_type)
    groups: dict[tuple[Any, ...], list[Node]] = {}
    for node in nodes:
        groups.setdefault(_memory_key(node), []).append(node)
    return [group for group in groups.values() if len(group) > 1]


def _memory_key(node: Node) -> tuple[Any, ...]:
    subject = str(node.content.get("subject", ""))
    text = str(node.content.get("text", ""))
    other_content = tuple(
        sorted(
            (key, _normalize_memory_text(str(value)))
            for key, value in node.content.items()
            if key not in {"text", "subject"} and isinstance(value, str)
        )
    )
    return (
        node.type,
        _normalize_memory_text(subject),
        _normalize_memory_text(text),
        other_content,
    )


def _format_memory_node(node: Node) -> str:
    text = str(node.content.get("text", "")).strip()
    subject = str(node.content.get("subject", "")).strip()
    subject_text = f" subject={subject!r}" if subject else ""
    return (
        f"[{node.type}] id={node.id}{subject_text} "
        f"source={node.source} confidence={node.confidence:.2f}: {text[:120]}"
    )
