"""Composable view builders for querying and writing to the memory graph."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from ai_arch_toolkit.toolkit.memory._types import (
    Edge,
    Node,
    NodeID,
    NodeType,
    SearchResult,
    _now_utc,
)
from ai_arch_toolkit.toolkit.memory.graph._store import GraphStore


class TemporalView:
    """Time-based queries and sequential writes."""

    __slots__ = ("_node_type", "_store")

    def __init__(self, store: GraphStore, *, node_type: NodeType | None = None) -> None:
        self._store = store
        self._node_type = node_type

    async def recent(self, *, k: int = 10) -> Sequence[Node]:
        """Get the k most recent nodes by timestamp."""
        nodes = await self._store.list(type=self._node_type)
        sorted_nodes = sorted(nodes, key=lambda n: n.timestamp, reverse=True)
        return sorted_nodes[:k]

    async def since(
        self, *, hours: float | None = None, minutes: float | None = None
    ) -> Sequence[Node]:
        """Get nodes since a relative time offset."""
        delta = timedelta(hours=hours or 0, minutes=minutes or 0)
        cutoff = _now_utc() - delta
        nodes = await self._store.list(type=self._node_type)
        return [n for n in nodes if n.timestamp >= cutoff]

    async def between(self, start: datetime, end: datetime) -> Sequence[Node]:
        """Get nodes with timestamp between start and end (event time)."""
        nodes = await self._store.list(type=self._node_type)
        return [n for n in nodes if start <= n.timestamp <= end]

    async def append(
        self,
        content: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
        source: str = "unknown",
        link_previous: bool = True,
    ) -> Node:
        """Append a new node, optionally linking to the most recent one."""
        node = Node(
            type=self._node_type or "generic",
            content=content,
            metadata=metadata or {},
            source=source,
        )
        node = await self._store.add(node)
        if link_previous:
            recent = await self.recent(k=2)
            # Link to the previous node (not the one we just added)
            for prev in recent:
                if prev.id != node.id:
                    await self._store.connect(prev.id, node.id, "NEXT")
                    break
        return node


class SimilarityView:
    """Vector similarity queries."""

    __slots__ = ("_node_type", "_store")

    def __init__(self, store: GraphStore, *, node_type: NodeType | None = None) -> None:
        self._store = store
        self._node_type = node_type

    async def find(self, query: str, *, k: int = 5) -> Sequence[SearchResult]:
        """Find nodes similar to a text query."""
        return await self._store.search(query, type=self._node_type, k=k)

    async def similar_to(self, node_id: NodeID, *, k: int = 5) -> Sequence[SearchResult]:
        """Find nodes similar to an existing node."""
        node = await self._store.backend.get_node(node_id)
        if node is None:
            return []
        # Use node content as query text
        text_parts = [str(v) for v in node.content.values() if isinstance(v, str)]
        query = " ".join(text_parts)
        if not query:
            return []
        results = await self._store.search(query, type=self._node_type, k=k + 1)
        # Exclude the source node itself
        return [r for r in results if r.node.id != node_id][:k]


class RelationalView:
    """Graph traversal and relationship queries."""

    __slots__ = ("_node_type", "_store")

    def __init__(self, store: GraphStore, *, node_type: NodeType | None = None) -> None:
        self._store = store
        self._node_type = node_type

    async def neighbors(
        self, node_id: NodeID, *, depth: int = 1, relation: str | None = None
    ) -> Sequence[Node]:
        """Get neighboring nodes."""
        nodes = await self._store.neighbors(node_id, depth=depth, relation=relation)
        if self._node_type is not None:
            return [n for n in nodes if n.type == self._node_type]
        return nodes

    async def path(self, source: NodeID, target: NodeID) -> Sequence[Node] | None:
        """Find shortest path between two nodes (requires algorithms support)."""
        if not self._store.has_algorithms:
            return None
        backend = self._store.backend
        return await backend.shortest_path(source, target)  # type: ignore[union-attr]

    async def edges(
        self,
        node_id: NodeID,
        *,
        direction: str = "out",
        relation: str | None = None,
    ) -> Sequence[Edge]:
        return await self._store.edges(node_id, direction=direction, relation=relation)

    async def connect(self, source: NodeID, target: NodeID, relation: str, **kw: Any) -> Edge:
        return await self._store.connect(source, target, relation, **kw)

    async def disconnect(self, source: NodeID, target: NodeID, relation: str) -> bool:
        return await self._store.disconnect(source, target, relation)


class PropertyView:
    """Metadata and lifecycle-based queries."""

    __slots__ = ("_node_type", "_store")

    def __init__(self, store: GraphStore, *, node_type: NodeType | None = None) -> None:
        self._store = store
        self._node_type = node_type

    async def filter(self, **metadata_filters: Any) -> Sequence[Node]:
        """Filter nodes by metadata key-value pairs."""
        nodes = await self._store.list(type=self._node_type)
        results: list[Node] = []
        for node in nodes:
            if all(node.metadata.get(k) == v for k, v in metadata_filters.items()):
                results.append(node)
        return results

    async def by_confidence(self, *, min_confidence: float = 0.5) -> Sequence[Node]:
        """Get nodes meeting a minimum confidence threshold."""
        nodes = await self._store.list(type=self._node_type)
        return [n for n in nodes if n.confidence >= min_confidence]

    async def by_source(self, source: str) -> Sequence[Node]:
        """Get nodes from a specific source."""
        nodes = await self._store.list(type=self._node_type)
        return [n for n in nodes if n.source == source]

    async def most_accessed(self, *, k: int = 10) -> Sequence[Node]:
        """Get the most frequently accessed nodes."""
        nodes = await self._store.list(type=self._node_type)
        sorted_nodes = sorted(nodes, key=lambda n: n.access_count, reverse=True)
        return sorted_nodes[:k]

    async def least_accessed(self, *, k: int = 10) -> Sequence[Node]:
        """Get the least frequently accessed nodes."""
        nodes = await self._store.list(type=self._node_type)
        sorted_nodes = sorted(nodes, key=lambda n: n.access_count)
        return sorted_nodes[:k]


def composite_score(
    result: SearchResult,
    *,
    similarity_weight: float = 0.5,
    recency_weight: float = 0.3,
    importance_weight: float = 0.2,
    recency_half_life_hours: float = 168,
) -> float:
    """Compute composite score combining similarity, recency, and importance.

    Recency decays exponentially with configurable half-life (default 1 week).
    Importance is derived from access_count (log-normalized).

    Args:
        result: Search result with node and similarity score.
        similarity_weight: Weight for the similarity component.
        recency_weight: Weight for the recency component.
        importance_weight: Weight for the importance/access component.
        recency_half_life_hours: Hours until recency score halves.
    """
    # Similarity component
    sim = max(0.0, min(1.0, result.score))

    # Recency component (exponential decay)
    age_hours = (_now_utc() - result.node.timestamp).total_seconds() / 3600
    decay = math.exp(-0.693 * age_hours / recency_half_life_hours)  # ln(2) ≈ 0.693

    # Importance component (log-normalized access count)
    importance = math.log1p(result.node.access_count) / (math.log1p(result.node.access_count) + 1)

    return similarity_weight * sim + recency_weight * decay + importance_weight * importance
