"""Graph — primary facade for general-purpose graph operations."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

from ai_arch_toolkit.core._persistence import load_json_object
from ai_arch_toolkit.core._sync import _run_sync
from ai_arch_toolkit.core.graph._backends import GraphAlgorithms, GraphBackend
from ai_arch_toolkit.core.graph._facade import GraphFacade, check_payload, edges_from
from ai_arch_toolkit.core.graph._types import Direction, Edge, Node, NodeID, NodeType

_GRAPH_SCHEMA_VERSION = 1


class Graph(GraphFacade[Node[Any]]):
    """General-purpose graph facade with type indexing, persistence, and sync wrappers.

    Delegates storage to a ``GraphBackend`` and optionally exposes graph
    algorithms when the backend implements ``GraphAlgorithms``. The node and edge surface is
    :class:`GraphFacade`'s, shared with the memory ``GraphStore``.
    """

    __slots__ = ("_backend",)

    _schema_version: ClassVar[int] = _GRAPH_SCHEMA_VERSION

    def __init__(self, backend: GraphBackend) -> None:
        super().__init__(backend)
        self._backend = backend

    # --- Properties ---

    @property
    def backend(self) -> GraphBackend:
        return self._backend

    # --- Async node ops ---

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

    # --- Async persistence ---

    def _node_record(self, node: Node[Any]) -> dict[str, Any]:
        # A dataclass content is saved as its fields; anything else as it is.
        content = node.content
        if dataclasses.is_dataclass(content) and not isinstance(content, type):
            content = dataclasses.asdict(content)
        return {"id": node.id, "type": node.type, "content": content, "metadata": node.metadata}

    @classmethod
    async def from_dict(
        cls,
        data: dict[str, Any],
        backend: GraphBackend,
        *,
        content_loader: Callable[[Any], Any] | None = None,
    ) -> Graph:
        """Deserialize a dict into a Graph."""
        check_payload(data, "Graph", _GRAPH_SCHEMA_VERSION, _check_graph_node)
        graph = cls(backend)
        load = content_loader or (lambda content: content)
        nodes = [  # all built first: a failing content_loader leaves the backend untouched
            Node(
                id=nd["id"],
                type=nd.get("type", "default"),
                content=load(nd.get("content")),
                metadata=nd.get("metadata", {}),
            )
            for nd in data["nodes"]
        ]
        await graph.add_many(nodes)
        for edge in edges_from(data["edges"]):
            await backend.add_edge(edge)
        return graph

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


def _check_graph_node(index: int, node: dict[str, Any]) -> None:
    if "id" not in node:
        raise ValueError(f"Graph node at index {index} missing required field 'id'")
    if "metadata" in node and not isinstance(node["metadata"], dict):
        raise ValueError(f"Graph node {node['id']!r} metadata must be an object")
