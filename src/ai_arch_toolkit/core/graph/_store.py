"""Graph — primary facade for general-purpose graph operations."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ai_arch_toolkit.core._persistence import atomic_write_json, load_json_object
from ai_arch_toolkit.core._sync import _run_sync
from ai_arch_toolkit.core.graph._backends import GraphAlgorithms, GraphBackend
from ai_arch_toolkit.core.graph._types import Direction, Edge, Node, NodeID, NodeType

_GRAPH_SCHEMA_VERSION = 1


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

    async def has(self, node_id: NodeID) -> bool:
        """Check if a node exists."""
        return await self._backend.get_node(node_id) is not None

    async def degree(self, node_id: NodeID) -> int:
        """Get the degree (in + out) of a node."""
        edges = await self._backend.get_edges(node_id, direction="both")
        return len(edges)

    async def node_count(self) -> int:
        """Count all nodes."""
        return await self._backend.count_nodes()

    async def edge_count(self) -> int:
        """Count all edges."""
        all_edges = await self._list_all_edges()
        return len(all_edges)

    async def is_empty(self) -> bool:
        """Check if the graph has no nodes."""
        return await self._backend.count_nodes() == 0

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

    async def get_edges_between(
        self, source: NodeID, target: NodeID, *, relation: str | None = None
    ) -> Sequence[Edge]:
        """Get all edges between two specific nodes."""
        out_edges = await self._backend.get_edges(source, direction="out", relation=relation)
        return [e for e in out_edges if e.target == target]

    async def list_edges(self, *, relation: str | None = None) -> Sequence[Edge]:
        """List all edges in the graph, optionally filtered by relation."""
        return await self._list_all_edges(relation=relation)

    # --- Internal helpers ---

    async def _list_all_edges(self, *, relation: str | None = None) -> list[Edge]:
        """Collect all outgoing edges from every node."""
        nodes = await self._backend.list_nodes()
        result: list[Edge] = []
        for n in nodes:
            result.extend(await self._backend.get_edges(n.id, direction="out", relation=relation))
        return result

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

    async def find_all_paths(
        self, source: NodeID, target: NodeID, *, max_depth: int | None = None
    ) -> Sequence[Sequence[NodeID]]:
        """Find all simple paths between two nodes."""
        if not isinstance(self._backend, GraphAlgorithms):
            msg = "Backend does not implement GraphAlgorithms"
            raise TypeError(msg)
        return await self._backend.find_all_paths(source, target, max_depth=max_depth)

    async def get_ancestors(self, node_id: NodeID) -> set[NodeID]:
        """Get all ancestors of a node (nodes that can reach it)."""
        if not isinstance(self._backend, GraphAlgorithms):
            msg = "Backend does not implement GraphAlgorithms"
            raise TypeError(msg)
        return await self._backend.ancestors(node_id)

    async def get_descendants(self, node_id: NodeID) -> set[NodeID]:
        """Get all descendants of a node (nodes reachable from it)."""
        if not isinstance(self._backend, GraphAlgorithms):
            msg = "Backend does not implement GraphAlgorithms"
            raise TypeError(msg)
        return await self._backend.descendants(node_id)

    async def get_subgraph(self, node_ids: Sequence[NodeID]) -> Graph:
        """Extract a subgraph containing only the specified nodes."""
        if not isinstance(self._backend, GraphAlgorithms):
            msg = "Backend does not implement GraphAlgorithms"
            raise TypeError(msg)
        sub_backend = await self._backend.subgraph(node_ids)
        return Graph(sub_backend)

    async def get_ego_graph(self, node_id: NodeID, *, radius: int = 1) -> Graph:
        """Get the ego graph (neighborhood) of a node."""
        if not isinstance(self._backend, GraphAlgorithms):
            msg = "Backend does not implement GraphAlgorithms"
            raise TypeError(msg)
        ego_backend = await self._backend.ego_graph(node_id, radius=radius)
        return Graph(ego_backend)

    async def pagerank(self, *, alpha: float = 0.85) -> dict[NodeID, float]:
        """Compute PageRank scores for all nodes."""
        if not isinstance(self._backend, GraphAlgorithms):
            msg = "Backend does not implement GraphAlgorithms"
            raise TypeError(msg)
        return await self._backend.pagerank(alpha=alpha)

    # --- Facade-only methods ---

    async def get_orphan_nodes(self) -> Sequence[Node[Any]]:
        """Get nodes with no edges (degree 0)."""
        nodes = await self._backend.list_nodes()
        # Build connected set in one pass over edges instead of per-node degree calls
        edges = await self._list_all_edges()
        connected: set[NodeID] = set()
        for e in edges:
            connected.add(e.source)
            connected.add(e.target)
        return [n for n in nodes if n.id not in connected]

    async def get_stats(self) -> dict[str, Any]:
        """Get summary statistics about the graph."""
        nodes = await self._backend.list_nodes()
        type_counts: dict[str, int] = {}
        for n in nodes:
            type_counts[n.type] = type_counts.get(n.type, 0) + 1
        edges = await self._list_all_edges()
        relation_counts: dict[str, int] = {}
        for e in edges:
            relation_counts[e.relation] = relation_counts.get(e.relation, 0) + 1
        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_types": type_counts,
            "edge_relations": relation_counts,
        }

    async def filter_nodes(self, predicate: Callable[[Node[Any]], bool]) -> Sequence[Node[Any]]:
        """Filter nodes by a predicate function."""
        nodes = await self._backend.list_nodes()
        return [n for n in nodes if predicate(n)]

    async def filter_edges(self, predicate: Callable[[Edge], bool]) -> Sequence[Edge]:
        """Filter edges by a predicate function."""
        edges = await self._list_all_edges()
        return [e for e in edges if predicate(e)]

    async def copy(self, backend: GraphBackend | None = None) -> Graph:
        """Create a deep copy of the graph.

        Args:
            backend: Backend for the copy. Required for backends with constructor
                arguments. Defaults to creating a new instance of the same type
                (only works for zero-arg constructors).
        """
        data = await self.to_dict()
        if backend is None:
            backend = type(self._backend)()
        return await Graph.from_dict(data, backend)

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
        return {
            "schema_version": _GRAPH_SCHEMA_VERSION,
            "nodes": nodes_data,
            "edges": edges_data,
        }

    @classmethod
    async def from_dict(
        cls,
        data: dict[str, Any],
        backend: GraphBackend,
        *,
        content_loader: Callable[[Any], Any] | None = None,
    ) -> Graph:
        """Deserialize a dict into a Graph."""
        _validate_graph_payload(data)
        graph = cls(backend)
        nodes: list[Node[Any]] = []
        edges: list[Edge] = []
        for nd in data["nodes"]:
            content = nd.get("content")
            if content_loader is not None:
                content = content_loader(content)
            nodes.append(
                Node(
                    id=nd["id"],
                    type=nd.get("type", "default"),
                    content=content,
                    metadata=nd.get("metadata", {}),
                )
            )
        for ed in data["edges"]:
            edges.append(
                Edge(
                    source=ed["source"],
                    target=ed["target"],
                    relation=ed["relation"],
                    weight=ed.get("weight", 1.0),
                    metadata=ed.get("metadata", {}),
                )
            )

        for node in nodes:
            await graph.add(node)
        for edge in edges:
            await backend.add_edge(edge)
        return graph

    async def save(self, path: str | Path) -> None:
        """Save the graph to a JSON file."""
        data = await self.to_dict()
        atomic_write_json(path, data)

    @classmethod
    async def load(
        cls,
        path: str | Path,
        backend: GraphBackend,
        *,
        content_loader: Callable[[Any], Any] | None = None,
    ) -> Graph:
        """Load a graph from a JSON file."""
        data = load_json_object(path)
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

    def has_sync(self, node_id: NodeID) -> bool:
        return _run_sync(self.has(node_id))

    def degree_sync(self, node_id: NodeID) -> int:
        return _run_sync(self.degree(node_id))

    def get_edges_between_sync(
        self, source: NodeID, target: NodeID, *, relation: str | None = None
    ) -> Sequence[Edge]:
        return _run_sync(self.get_edges_between(source, target, relation=relation))

    def list_edges_sync(self, *, relation: str | None = None) -> Sequence[Edge]:
        return _run_sync(self.list_edges(relation=relation))

    def node_count_sync(self) -> int:
        return _run_sync(self.node_count())

    def edge_count_sync(self) -> int:
        return _run_sync(self.edge_count())

    def is_empty_sync(self) -> bool:
        return _run_sync(self.is_empty())

    def get_orphan_nodes_sync(self) -> Sequence[Node[Any]]:
        return _run_sync(self.get_orphan_nodes())

    def get_stats_sync(self) -> dict[str, Any]:
        return _run_sync(self.get_stats())

    def filter_nodes_sync(self, predicate: Callable[[Node[Any]], bool]) -> Sequence[Node[Any]]:
        return _run_sync(self.filter_nodes(predicate))

    def filter_edges_sync(self, predicate: Callable[[Edge], bool]) -> Sequence[Edge]:
        return _run_sync(self.filter_edges(predicate))

    def copy_sync(self, backend: GraphBackend | None = None) -> Graph:
        return _run_sync(self.copy(backend))

    def find_all_paths_sync(
        self, source: NodeID, target: NodeID, *, max_depth: int | None = None
    ) -> Sequence[Sequence[NodeID]]:
        return _run_sync(self.find_all_paths(source, target, max_depth=max_depth))

    def get_ancestors_sync(self, node_id: NodeID) -> set[NodeID]:
        return _run_sync(self.get_ancestors(node_id))

    def get_descendants_sync(self, node_id: NodeID) -> set[NodeID]:
        return _run_sync(self.get_descendants(node_id))

    def get_subgraph_sync(self, node_ids: Sequence[NodeID]) -> Graph:
        return _run_sync(self.get_subgraph(node_ids))

    def get_ego_graph_sync(self, node_id: NodeID, *, radius: int = 1) -> Graph:
        return _run_sync(self.get_ego_graph(node_id, radius=radius))

    def pagerank_sync(self, *, alpha: float = 0.85) -> dict[NodeID, float]:
        return _run_sync(self.pagerank(alpha=alpha))

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


def _validate_graph_payload(data: dict[str, Any]) -> None:
    version = data.get("schema_version", 0)
    if not isinstance(version, int):
        raise ValueError("Graph payload schema_version must be an integer")
    if version > _GRAPH_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported graph schema_version {version}; "
            f"maximum supported is {_GRAPH_SCHEMA_VERSION}"
        )
    if version < 0:
        raise ValueError(f"Unsupported graph schema_version {version}")

    nodes = data.get("nodes")
    edges = data.get("edges")
    if not isinstance(nodes, list):
        raise ValueError("Graph payload must contain a 'nodes' list")
    if not isinstance(edges, list):
        raise ValueError("Graph payload must contain an 'edges' list")

    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"Graph node at index {index} must be an object")
        if "id" not in node:
            raise ValueError(f"Graph node at index {index} missing required field 'id'")
        if "metadata" in node and not isinstance(node["metadata"], dict):
            raise ValueError(f"Graph node {node['id']!r} metadata must be an object")

    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise ValueError(f"Graph edge at index {index} must be an object")
        for field in ("source", "target", "relation"):
            if field not in edge:
                raise ValueError(f"Graph edge at index {index} missing required field {field!r}")
        if "metadata" in edge and not isinstance(edge["metadata"], dict):
            raise ValueError(f"Graph edge at index {index} metadata must be an object")
