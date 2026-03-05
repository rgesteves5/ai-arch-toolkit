"""GraphStore — primary facade with auto-embedding, lifecycle tracking, and persistence."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_arch_toolkit.core.graph import Direction
from ai_arch_toolkit.toolkit.memory._types import (
    Edge,
    EmbedFn,
    Node,
    NodeID,
    NodeType,
    SearchResult,
    _now_utc,
)
from ai_arch_toolkit.toolkit.memory.graph._backends import (
    GraphAlgorithms,
    GraphBackend,
    MemoryBackend,
)
from ai_arch_toolkit.toolkit.memory.graph._index import BruteForceIndex, VectorIndex


def _embeddable_text(node: Node) -> str:
    """Extract embeddable text from node content (string values only)."""
    parts = [str(v) for v in node.content.values() if isinstance(v, str)]
    return " ".join(parts)


class GraphStore:
    """Primary interface for the memory graph.

    Coordinates a graph backend, optional vector index, and optional embedding
    function. Handles auto-embedding, access tracking, type indexing, and
    persistence.
    """

    __slots__ = ("_backend", "_embed", "_index", "_type_index")

    def __init__(
        self,
        backend: GraphBackend & MemoryBackend,
        *,
        embed: EmbedFn | None = None,
        index: VectorIndex | None = None,
    ) -> None:
        self._backend: GraphBackend & MemoryBackend = backend
        self._embed = embed
        self._index: VectorIndex | None = index
        if embed is not None and index is None:
            self._index = BruteForceIndex()
        self._type_index: dict[str, set[NodeID]] = {}

    # --- Properties ---

    @property
    def backend(self) -> GraphBackend:
        return self._backend

    @property
    def has_embeddings(self) -> bool:
        return self._embed is not None

    @property
    def has_algorithms(self) -> bool:
        return isinstance(self._backend, GraphAlgorithms)

    # --- Node operations ---

    async def add(self, node: Node) -> Node:
        """Add a node, auto-embedding if an embed function is configured."""
        if self._embed is not None and node.embedding is None:
            text = _embeddable_text(node)
            if text.strip():
                embedding = await self._embed(text)
                node = dataclasses.replace(node, embedding=embedding)
        await self._backend.add_node(node)
        if node.embedding is not None and self._index is not None:
            await self._index.add(node.id, node.embedding)
        self._type_index.setdefault(node.type, set()).add(node.id)
        return node

    async def get(self, node_id: NodeID) -> Node | None:
        """Get a node by ID, bumping access tracking."""
        node = await self._backend.get_node(node_id)
        if node is None:
            return None
        updated = await self._backend.update_node(
            node_id,
            access_count=node.access_count + 1,
            last_accessed=_now_utc(),
        )
        return updated or node

    async def update(self, node_id: NodeID, **attrs: Any) -> Node | None:
        """Update node attributes. Re-embeds if content changes."""
        old = await self._backend.get_node(node_id)
        if old is None:
            return None
        updated = await self._backend.update_node(node_id, **attrs)
        if updated is None:
            return None
        # Re-embed if content changed
        if "content" in attrs and self._embed is not None:
            text = _embeddable_text(updated)
            if text.strip():
                embedding = await self._embed(text)
                updated = await self._backend.update_node(node_id, embedding=embedding)
                if updated is not None and self._index is not None:
                    await self._index.update(node_id, updated.embedding or embedding)
            elif old.embedding is not None:
                # Content no longer has embeddable text — clear stale embedding
                updated = await self._backend.update_node(node_id, embedding=None)
                if self._index is not None:
                    await self._index.remove(node_id)
        # Update type index if type changed
        if "type" in attrs:
            old_type_set = self._type_index.get(old.type)
            if old_type_set:
                old_type_set.discard(node_id)
            self._type_index.setdefault(updated.type, set()).add(node_id)
        return updated

    async def remove(self, node_id: NodeID) -> bool:
        """Remove a node and clean up index entries."""
        node = await self._backend.get_node(node_id)
        if node is None:
            return False
        removed = await self._backend.remove_node(node_id)
        if removed:
            # Update type index first (pure dict op, won't fail)
            type_set = self._type_index.get(node.type)
            if type_set:
                type_set.discard(node_id)
            # Then update vector index (may involve I/O)
            if self._index is not None:
                await self._index.remove(node_id)
        return removed

    async def list(
        self, *, type: NodeType | None = None, limit: int | None = None
    ) -> Sequence[Node]:
        """List nodes, using type index for O(k) lookup when available."""
        if type is not None and type in self._type_index:
            ids = self._type_index[type]
            nodes: list[Node] = []
            for nid in ids:
                node = await self._backend.get_node(nid)
                if node is not None:
                    nodes.append(node)
                    if limit is not None and len(nodes) >= limit:
                        break
            return nodes
        return await self._backend.list_nodes(type=type, limit=limit)

    async def count(self, *, type: NodeType | None = None) -> int:
        """Count nodes, using type index when available."""
        if type is not None and type in self._type_index:
            return len(self._type_index[type])
        return await self._backend.count_nodes(type=type)

    # --- Edge operations ---

    async def connect(
        self,
        source: NodeID,
        target: NodeID,
        relation: str,
        *,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> Edge:
        """Create an edge between two nodes."""
        edge = Edge(
            source=source,
            target=target,
            relation=relation,
            weight=weight,
            metadata=metadata or {},
        )
        await self._backend.add_edge(edge)
        return edge

    async def edges(
        self,
        node_id: NodeID,
        *,
        direction: Direction = "out",
        relation: str | None = None,
    ) -> Sequence[Edge]:
        return await self._backend.get_edges(node_id, direction=direction, relation=relation)

    async def disconnect(self, source: NodeID, target: NodeID, relation: str) -> bool:
        return await self._backend.remove_edge(source, target, relation)

    # --- Search ---

    async def search(
        self, query: str, *, type: NodeType | None = None, k: int = 5
    ) -> Sequence[SearchResult]:
        """Search for nodes matching a query.

        Uses vector similarity when an embed function is configured, falling
        back to keyword search. The search cascade:

        1. If embed fn: embed query, try backend.search_similar (native vector)
        2. If backend returns None: try index.search (BruteForce/Faiss/etc.)
        3. If no embed fn: backend.search_content (keyword)
        """
        if self._embed is not None:
            query_embedding = await self._embed(query)
            # Try native vector search first
            native = await self._backend.search_similar(query_embedding, type=type, k=k)
            if native is not None:
                return [SearchResult(node=n, score=1.0) for n in native]
            # Fall back to index
            if self._index is not None:
                pairs = await self._index.search(query_embedding, k=k * 2)
                results: list[SearchResult] = []
                for nid, score in pairs:
                    node = await self._backend.get_node(nid)
                    if node is None:
                        continue
                    if type is not None and node.type != type:
                        continue
                    results.append(SearchResult(node=node, score=score))
                    if len(results) >= k:
                        break
                return results
        # Keyword fallback
        nodes = await self._backend.search_content(query, type=type, k=k)
        return [SearchResult(node=n, score=1.0) for n in nodes]

    # --- Traversal ---

    async def neighbors(
        self, node_id: NodeID, *, depth: int = 1, relation: str | None = None
    ) -> Sequence[Node]:
        return await self._backend.neighbors(node_id, depth=depth, relation=relation)

    # --- Bulk ---

    async def clear(self, *, type: NodeType | None = None) -> int:
        """Clear nodes. Updates type index and vector index accordingly."""
        # Snapshot IDs to remove from vector index before clearing anything
        if type is None:
            ids_to_remove = [nid for ids in self._type_index.values() for nid in ids]
        else:
            ids_to_remove = list(self._type_index.get(type, set()))
        # Clear backend and type index first (these must stay consistent)
        count = await self._backend.clear(type=type)
        if type is None:
            self._type_index.clear()
        else:
            self._type_index.pop(type, None)
        # Then clean up vector index (best-effort, won't desync core state)
        if self._index is not None:
            for nid in ids_to_remove:
                await self._index.remove(nid)
        return count

    async def add_many(self, nodes: Sequence[Node]) -> int:
        """Add multiple nodes. Returns count added."""
        count = 0
        for node in nodes:
            await self.add(node)
            count += 1
        return count

    async def remove_many(self, node_ids: Sequence[NodeID]) -> int:
        """Remove multiple nodes. Returns count removed."""
        count = 0
        for nid in node_ids:
            if await self.remove(nid):
                count += 1
        return count

    # --- Persistence ---

    async def to_dict(self) -> dict[str, Any]:
        """Serialize all nodes and edges to a dict."""
        all_nodes = await self._backend.list_nodes()
        nodes_data: list[dict[str, Any]] = []
        edges_data: list[dict[str, Any]] = []
        for node in all_nodes:
            nd: dict[str, Any] = {
                "id": node.id,
                "type": node.type,
                "content": node.content,
                "metadata": node.metadata,
                "embedding": node.embedding,
                "timestamp": node.timestamp.isoformat(),
                "created_at": node.created_at.isoformat(),
                "access_count": node.access_count,
                "last_accessed": (node.last_accessed.isoformat() if node.last_accessed else None),
                "confidence": node.confidence,
                "source": node.source,
            }
            nodes_data.append(nd)
            for edge in await self._backend.get_edges(node.id, direction="out"):
                edges_data.append(
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "relation": edge.relation,
                        "weight": edge.weight,
                        "metadata": edge.metadata,
                    }
                )
        return {"nodes": nodes_data, "edges": edges_data}

    @classmethod
    async def from_dict(
        cls,
        data: dict[str, Any],
        backend: GraphBackend,
        *,
        embed: EmbedFn | None = None,
        index: VectorIndex | None = None,
    ) -> GraphStore:
        """Deserialize a dict into a GraphStore."""
        store = cls(backend, embed=embed, index=index)
        for nd in data.get("nodes", []):
            last = nd.get("last_accessed")
            node = Node(
                id=nd["id"],
                type=nd.get("type", "generic"),
                content=nd.get("content", {}),
                metadata=nd.get("metadata", {}),
                embedding=nd.get("embedding"),
                timestamp=datetime.fromisoformat(nd["timestamp"]),
                created_at=datetime.fromisoformat(nd["created_at"]),
                access_count=nd.get("access_count", 0),
                last_accessed=datetime.fromisoformat(last) if last else None,
                confidence=nd.get("confidence", 1.0),
                source=nd.get("source", "unknown"),
            )
            # Add directly to backend to preserve original state (no re-embedding)
            await backend.add_node(node)
            if node.embedding is not None and store._index is not None:
                await store._index.add(node.id, node.embedding)
            store._type_index.setdefault(node.type, set()).add(node.id)
        for ed in data.get("edges", []):
            edge = Edge(
                source=ed["source"],
                target=ed["target"],
                relation=ed["relation"],
                weight=ed.get("weight", 1.0),
                metadata=ed.get("metadata", {}),
            )
            await backend.add_edge(edge)
        return store

    async def save(self, path: str | Path) -> None:
        """Save the graph to a JSON file."""
        data = await self.to_dict()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    async def load(
        cls,
        path: str | Path,
        backend: GraphBackend,
        *,
        embed: EmbedFn | None = None,
        index: VectorIndex | None = None,
    ) -> GraphStore:
        """Load a graph from a JSON file."""
        p = Path(path)
        data = json.loads(p.read_text())
        return await cls.from_dict(data, backend, embed=embed, index=index)
