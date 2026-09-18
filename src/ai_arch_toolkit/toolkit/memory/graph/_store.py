"""GraphStore — the memory graph: auto-embedding, access tracking, vector search, persistence.

The node and edge surface is the core graph's (``GraphFacade``, shared with ``Graph``); this module
adds only what is the memory's own.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from ai_arch_toolkit.core._persistence import load_json_object
from ai_arch_toolkit.core.graph._facade import GraphFacade, check_payload, edges_from
from ai_arch_toolkit.toolkit.memory._types import (
    EmbedFn,
    Node,
    NodeID,
    NodeType,
    SearchResult,
    _now_utc,
)
from ai_arch_toolkit.toolkit.memory.graph._backends import MemoryBackend
from ai_arch_toolkit.toolkit.memory.graph._index import BruteForceIndex, VectorIndex

_MEMORY_SCHEMA_VERSION = 1


def _embeddable_text(node: Node) -> str:
    """Extract embeddable text from node content (string values only)."""
    parts = [str(v) for v in node.content.values() if isinstance(v, str)]
    return " ".join(parts)


class GraphStore(GraphFacade[Node]):
    """Primary interface for the memory graph.

    Coordinates a graph backend, optional vector index, and optional embedding
    function. Handles auto-embedding, access tracking, type indexing, and
    persistence.
    """

    __slots__ = ("_backend", "_embed", "_index")

    _schema_version: ClassVar[int] = _MEMORY_SCHEMA_VERSION

    def __init__(
        self,
        backend: MemoryBackend,
        *,
        embed: EmbedFn | None = None,
        index: VectorIndex | None = None,
    ) -> None:
        super().__init__(backend)
        self._backend: MemoryBackend = backend
        self._embed = embed
        self._index: VectorIndex | None = index
        if embed is not None and index is None:
            self._index = BruteForceIndex()

    # --- Properties ---

    @property
    def backend(self) -> MemoryBackend:
        return self._backend

    @property
    def has_embeddings(self) -> bool:
        return self._embed is not None

    # --- Node operations ---

    async def add(self, node: Node) -> Node:
        """Add a node, auto-embedding if an embed function is configured."""
        if self._embed is not None and node.embedding is None:
            text = _embeddable_text(node)
            if text.strip():
                node = dataclasses.replace(node, embedding=await self._embed(text))
        await super().add(node)
        if node.embedding is not None and self._index is not None:
            await self._index.add(node.id, node.embedding)
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
        changed = await self._update(node_id, attrs)
        if changed is None:
            return None
        old, updated = changed
        if "content" not in attrs or self._embed is None:
            return updated
        return await self._reembed(updated, self._embed, stale=old.embedding is not None)

    async def _reembed(self, node: Node, embed: EmbedFn, *, stale: bool) -> Node:
        """``node`` embedded for its new content; without its old embedding if it has no text."""
        text = _embeddable_text(node)
        if text.strip():
            embedding = await embed(text)
            reembedded = await self._backend.update_node(node.id, embedding=embedding)
            if reembedded is None:
                return node
            if self._index is not None:
                await self._index.update(node.id, reembedded.embedding or embedding)
            return reembedded
        if not stale:
            return node
        cleared = await self._backend.update_node(node.id, embedding=None)
        if self._index is not None:
            await self._index.remove(node.id)
        return cleared or node

    async def remove(self, node_id: NodeID) -> bool:
        """Remove a node and clean up index entries."""
        removed = await super().remove(node_id)
        if removed and self._index is not None:
            await self._index.remove(node_id)
        return removed

    async def clear(self, *, type: NodeType | None = None) -> int:
        """Clear nodes. Updates type index and vector index accordingly."""
        ids = self._indexed_ids(type)  # before clearing: the vector index is cleaned after
        count = await super().clear(type=type)
        if self._index is not None:
            for node_id in ids:
                await self._index.remove(node_id)
        return count

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

    # --- Persistence ---

    def _node_record(self, node: Node) -> dict[str, Any]:
        return {
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

    @classmethod
    async def from_dict(
        cls,
        data: dict[str, Any],
        backend: MemoryBackend,
        *,
        embed: EmbedFn | None = None,
        index: VectorIndex | None = None,
    ) -> GraphStore:
        """Deserialize a dict into a GraphStore."""
        check_payload(data, "Memory graph", _MEMORY_SCHEMA_VERSION, _check_memory_node)
        store = cls(backend, embed=embed, index=index)
        for node in [_memory_node(record) for record in data["nodes"]]:
            # Into the graph as saved: the facade's own add, so nothing is re-embedded.
            await GraphFacade.add(store, node)
            if node.embedding is not None and store._index is not None:
                await store._index.add(node.id, node.embedding)
        for edge in edges_from(data["edges"]):
            await backend.add_edge(edge)
        return store

    @classmethod
    async def load(
        cls,
        path: str | Path,
        backend: MemoryBackend,
        *,
        embed: EmbedFn | None = None,
        index: VectorIndex | None = None,
    ) -> GraphStore:
        """Load a graph from a JSON file."""
        data = load_json_object(path)
        return await cls.from_dict(data, backend, embed=embed, index=index)


def _memory_node(record: dict[str, Any]) -> Node:
    last = record.get("last_accessed")
    return Node(
        id=record["id"],
        type=record.get("type", "generic"),
        content=record.get("content", {}),
        metadata=record.get("metadata", {}),
        embedding=record.get("embedding"),
        timestamp=datetime.fromisoformat(record["timestamp"]),
        created_at=datetime.fromisoformat(record["created_at"]),
        access_count=record.get("access_count", 0),
        last_accessed=datetime.fromisoformat(last) if last else None,
        confidence=record.get("confidence", 1.0),
        source=record.get("source", "unknown"),
    )


def _check_memory_node(index: int, node: dict[str, Any]) -> None:
    for field in ("id", "timestamp", "created_at"):
        if field not in node:
            msg = f"Memory graph node at index {index} missing required field {field!r}"
            raise ValueError(msg)
    for field in ("content", "metadata"):
        if field in node and not isinstance(node[field], dict):
            raise ValueError(f"Memory graph node {node['id']!r} {field} must be an object")
    try:
        datetime.fromisoformat(node["timestamp"])
        datetime.fromisoformat(node["created_at"])
        if node.get("last_accessed"):
            datetime.fromisoformat(node["last_accessed"])
    except ValueError as exc:
        raise ValueError(f"Memory graph node {node['id']!r} has invalid datetime field") from exc
