"""Graph — primary facade for general-purpose graph operations."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ai_arch_toolkit.core._sync import _run_sync
from ai_arch_toolkit.core.graph._backends import GraphAlgorithms, GraphBackend
from ai_arch_toolkit.core.graph._types import Direction, Edge, Node, NodeID, NodeType


class Graph:
    """General-purpose graph facade with type indexing, persistence, and sync wrappers.

    Delegates storage to a ``GraphBackend`` and optionally exposes graph
    algorithms when the backend implements ``GraphAlgorithms``.
    """

    __slots__ = ("_backend", "_type_index")

    def __init__(self, backend: GraphBackend) -> None:
        self._backend = backend
        self._type_index: dict[str, set[NodeID]] = {}

    # --- Properties ---

    @property
    def backend(self) -> GraphBackend:
        return self._backend

    @property
    def has_algorithms(self) -> bool:
        return isinstance(self._backend, GraphAlgorithms)

    # --- Async node ops ---

    async def add(self, node: Node[Any]) -> Node[Any]:
        """Add a node to the graph."""
        await self._backend.add_node(node)
        self._type_index.setdefault(node.type, set()).add(node.id)
        return node

    async def get(self, node_id: NodeID) -> Node[Any] | None:
        """Get a node by ID."""
        return await self._backend.get_node(node_id)

    async def update(self, node_id: NodeID, **attrs: Any) -> Node[Any] | None:
        """Update node attributes."""
        old = await self._backend.get_node(node_id)
        if old is None:
            return None
        updated = await self._backend.update_node(node_id, **attrs)
        if updated is None:
            return None
        if "type" in attrs:
            old_type_set = self._type_index.get(old.type)
            if old_type_set:
                old_type_set.discard(node_id)
            self._type_index.setdefault(updated.type, set()).add(node_id)
        return updated

    async def remove(self, node_id: NodeID) -> bool:
        """Remove a node."""
        node = await self._backend.get_node(node_id)
        if node is None:
            return False
        removed = await self._backend.remove_node(node_id)
        if removed:
            type_set = self._type_index.get(node.type)
            if type_set:
                type_set.discard(node_id)
        return removed

    async def list(
        self, *, type: NodeType | None = None, limit: int | None = None
    ) -> Sequence[Node[Any]]:
        """List nodes, using type index for O(k) lookup when available."""
        if type is not None and type in self._type_index:
            ids = self._type_index[type]
            nodes: list[Node[Any]] = []
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

    # --- Async edge ops ---

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

    # --- Async traversal + algorithms ---

    async def neighbors(
        self, node_id: NodeID, *, depth: int = 1, relation: str | None = None
    ) -> Sequence[Node[Any]]:
        return await self._backend.neighbors(node_id, depth=depth, relation=relation)

    async def bfs(self, start: NodeID, *, relation: str | None = None) -> Sequence[Node[Any]]:
        if not isinstance(self._backend, GraphAlgorithms):
            msg = "Backend does not implement GraphAlgorithms"
            raise TypeError(msg)
        return await self._backend.bfs(start, relation=relation)

    async def dfs(self, start: NodeID, *, relation: str | None = None) -> Sequence[Node[Any]]:
        if not isinstance(self._backend, GraphAlgorithms):
            msg = "Backend does not implement GraphAlgorithms"
            raise TypeError(msg)
        return await self._backend.dfs(start, relation=relation)

    async def shortest_path(
        self, source: NodeID, target: NodeID, *, relation: str | None = None
    ) -> Sequence[Node[Any]] | None:
        if not isinstance(self._backend, GraphAlgorithms):
            msg = "Backend does not implement GraphAlgorithms"
            raise TypeError(msg)
        return await self._backend.shortest_path(source, target, relation=relation)

    async def centrality(self, *, relation: str | None = None) -> dict[NodeID, float]:
        if not isinstance(self._backend, GraphAlgorithms):
            msg = "Backend does not implement GraphAlgorithms"
            raise TypeError(msg)
        return await self._backend.centrality(relation=relation)

    async def connected_components(
        self, *, relation: str | None = None
    ) -> Sequence[Sequence[NodeID]]:
        if not isinstance(self._backend, GraphAlgorithms):
            msg = "Backend does not implement GraphAlgorithms"
            raise TypeError(msg)
        return await self._backend.connected_components(relation=relation)

    # --- Async bulk ---

    async def clear(self, *, type: NodeType | None = None) -> int:
        """Clear nodes from the graph."""
        count = await self._backend.clear(type=type)
        if type is None:
            self._type_index.clear()
        else:
            self._type_index.pop(type, None)
        return count

    async def add_many(self, nodes: Sequence[Node[Any]]) -> int:
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

    # --- Async persistence ---

    async def to_dict(self) -> dict[str, Any]:
        """Serialize all nodes and edges to a dict."""
        all_nodes = await self._backend.list_nodes()
        nodes_data: list[dict[str, Any]] = []
        edges_data: list[dict[str, Any]] = []
        for node in all_nodes:
            # Serialize content: use asdict for dataclasses, pass through otherwise
            if dataclasses.is_dataclass(node.content) and not isinstance(node.content, type):
                content = dataclasses.asdict(node.content)
            else:
                content = node.content
            nd: dict[str, Any] = {
                "id": node.id,
                "type": node.type,
                "content": content,
                "metadata": node.metadata,
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
        content_loader: Callable[[Any], Any] | None = None,
    ) -> Graph:
        """Deserialize a dict into a Graph."""
        graph = cls(backend)
        for nd in data.get("nodes", []):
            content = nd.get("content")
            if content_loader is not None:
                content = content_loader(content)
            node: Node[Any] = Node(
                id=nd["id"],
                type=nd.get("type", "default"),
                content=content,
                metadata=nd.get("metadata", {}),
            )
            await graph.add(node)
        for ed in data.get("edges", []):
            edge = Edge(
                source=ed["source"],
                target=ed["target"],
                relation=ed["relation"],
                weight=ed.get("weight", 1.0),
                metadata=ed.get("metadata", {}),
            )
            await backend.add_edge(edge)
        return graph

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
        content_loader: Callable[[Any], Any] | None = None,
    ) -> Graph:
        """Load a graph from a JSON file."""
        p = Path(path)
        data = json.loads(p.read_text())
        return await cls.from_dict(data, backend, content_loader=content_loader)

    # --- Sync wrappers ---

    def add_sync(self, node: Node[Any]) -> Node[Any]:
        return _run_sync(self.add(node))

    def get_sync(self, node_id: NodeID) -> Node[Any] | None:
        return _run_sync(self.get(node_id))

    def update_sync(self, node_id: NodeID, **attrs: Any) -> Node[Any] | None:
        return _run_sync(self.update(node_id, **attrs))

    def remove_sync(self, node_id: NodeID) -> bool:
        return _run_sync(self.remove(node_id))

    def list_sync(
        self, *, type: NodeType | None = None, limit: int | None = None
    ) -> Sequence[Node[Any]]:
        return _run_sync(self.list(type=type, limit=limit))

    def count_sync(self, *, type: NodeType | None = None) -> int:
        return _run_sync(self.count(type=type))

    def connect_sync(
        self,
        source: NodeID,
        target: NodeID,
        relation: str,
        *,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> Edge:
        return _run_sync(self.connect(source, target, relation, weight=weight, metadata=metadata))

    def edges_sync(
        self,
        node_id: NodeID,
        *,
        direction: Direction = "out",
        relation: str | None = None,
    ) -> Sequence[Edge]:
        return _run_sync(self.edges(node_id, direction=direction, relation=relation))

    def disconnect_sync(self, source: NodeID, target: NodeID, relation: str) -> bool:
        return _run_sync(self.disconnect(source, target, relation))

    def neighbors_sync(
        self, node_id: NodeID, *, depth: int = 1, relation: str | None = None
    ) -> Sequence[Node[Any]]:
        return _run_sync(self.neighbors(node_id, depth=depth, relation=relation))

    def bfs_sync(self, start: NodeID, *, relation: str | None = None) -> Sequence[Node[Any]]:
        return _run_sync(self.bfs(start, relation=relation))

    def dfs_sync(self, start: NodeID, *, relation: str | None = None) -> Sequence[Node[Any]]:
        return _run_sync(self.dfs(start, relation=relation))

    def shortest_path_sync(
        self, source: NodeID, target: NodeID, *, relation: str | None = None
    ) -> Sequence[Node[Any]] | None:
        return _run_sync(self.shortest_path(source, target, relation=relation))

    def centrality_sync(self, *, relation: str | None = None) -> dict[NodeID, float]:
        return _run_sync(self.centrality(relation=relation))

    def connected_components_sync(
        self, *, relation: str | None = None
    ) -> Sequence[Sequence[NodeID]]:
        return _run_sync(self.connected_components(relation=relation))

    def clear_sync(self, *, type: NodeType | None = None) -> int:
        return _run_sync(self.clear(type=type))

    def add_many_sync(self, nodes: Sequence[Node[Any]]) -> int:
        return _run_sync(self.add_many(nodes))

    def remove_many_sync(self, node_ids: Sequence[NodeID]) -> int:
        return _run_sync(self.remove_many(node_ids))

    def save_sync(self, path: str | Path) -> None:
        return _run_sync(self.save(path))

    @classmethod
    def from_dict_sync(
        cls,
        data: dict[str, Any],
        backend: GraphBackend,
        *,
        content_loader: Callable[[Any], Any] | None = None,
    ) -> Graph:
        return _run_sync(cls.from_dict(data, backend, content_loader=content_loader))

    @classmethod
    def load_sync(
        cls,
        path: str | Path,
        backend: GraphBackend,
        *,
        content_loader: Callable[[Any], Any] | None = None,
    ) -> Graph:
        return _run_sync(cls.load(path, backend, content_loader=content_loader))
