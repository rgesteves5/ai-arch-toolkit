"""NetworkX-based graph backend — default in-memory implementation."""

from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Sequence
from typing import Any

from ai_arch_toolkit.core.graph._types import Direction, Edge, Node, NodeID, NodeType

try:
    import networkx as nx  # type: ignore[import-untyped]
except ImportError as e:
    msg = "NetworkXBackend requires networkx. Install it with: pip install ai-arch-toolkit[graph]"
    raise ImportError(msg) from e


class NetworkXBackend:
    """Async graph backend backed by NetworkX MultiDiGraph.

    Wraps sync NetworkX operations as async methods. Uses ``relation`` as the
    edge key to support multiple edges between the same pair of nodes.
    """

    __slots__ = ("_graph",)

    def __init__(self) -> None:
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()

    # --- Node operations ---

    async def add_node(self, node: Node[Any]) -> None:
        self._graph.add_node(node.id, node=node)

    async def get_node(self, node_id: NodeID) -> Node[Any] | None:
        data = self._graph.nodes.get(node_id)
        if data is None:
            return None
        return data.get("node")

    async def update_node(self, node_id: NodeID, **attrs: object) -> Node[Any] | None:
        data = self._graph.nodes.get(node_id)
        if data is None:
            return None
        old: Node[Any] = data["node"]
        updated = dataclasses.replace(old, **attrs)
        self._graph.nodes[node_id]["node"] = updated
        return updated

    async def remove_node(self, node_id: NodeID) -> bool:
        if node_id not in self._graph:
            return False
        self._graph.remove_node(node_id)
        return True

    async def list_nodes(
        self, *, type: NodeType | None = None, limit: int | None = None
    ) -> Sequence[Node[Any]]:
        nodes: list[Node[Any]] = []
        for _, data in self._graph.nodes(data=True):
            node: Node[Any] = data["node"]
            if type is not None and node.type != type:
                continue
            nodes.append(node)
            if limit is not None and len(nodes) >= limit:
                break
        return nodes

    async def count_nodes(self, *, type: NodeType | None = None) -> int:
        if type is None:
            return self._graph.number_of_nodes()
        return sum(1 for _, d in self._graph.nodes(data=True) if d["node"].type == type)

    # --- Edge operations ---

    async def add_edge(self, edge: Edge) -> None:
        if edge.source not in self._graph or edge.target not in self._graph:
            missing = [nid for nid in (edge.source, edge.target) if nid not in self._graph]
            msg = f"Cannot add edge — node(s) not found: {missing}"
            raise ValueError(msg)
        self._graph.add_edge(
            edge.source,
            edge.target,
            key=edge.relation,
            edge=edge,
        )

    async def get_edges(
        self,
        node_id: NodeID,
        *,
        direction: Direction = "out",
        relation: str | None = None,
    ) -> Sequence[Edge]:
        if node_id not in self._graph:
            return []
        edges: list[Edge] = []
        if direction in ("out", "both"):
            for _, _, _, data in self._graph.out_edges(node_id, data=True, keys=True):
                e: Edge = data["edge"]
                if relation is None or e.relation == relation:
                    edges.append(e)
        if direction in ("in", "both"):
            for _, _, _, data in self._graph.in_edges(node_id, data=True, keys=True):
                e = data["edge"]
                if relation is None or e.relation == relation:
                    edges.append(e)
        return edges

    async def remove_edge(self, source: NodeID, target: NodeID, relation: str) -> bool:
        if not self._graph.has_edge(source, target, key=relation):
            return False
        self._graph.remove_edge(source, target, key=relation)
        return True

    # --- Traversal ---

    async def neighbors(
        self, node_id: NodeID, *, depth: int = 1, relation: str | None = None
    ) -> Sequence[Node[Any]]:
        if node_id not in self._graph:
            return []
        visited: set[NodeID] = {node_id}
        queue: deque[tuple[NodeID, int]] = deque([(node_id, 0)])
        result: list[Node[Any]] = []
        while queue:
            current, d = queue.popleft()
            if d >= depth:
                continue
            for _, neighbor, _, data in self._graph.out_edges(current, data=True, keys=True):
                e: Edge = data["edge"]
                if relation is not None and e.relation != relation:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    node_data = self._graph.nodes[neighbor]
                    result.append(node_data["node"])
                    queue.append((neighbor, d + 1))
        return result

    # --- Bulk ---

    async def clear(self, *, type: NodeType | None = None) -> int:
        if type is None:
            count = self._graph.number_of_nodes()
            self._graph.clear()
            return count
        to_remove = [nid for nid, d in self._graph.nodes(data=True) if d["node"].type == type]
        for nid in to_remove:
            self._graph.remove_node(nid)
        return len(to_remove)

    # --- Extended queries ---

    async def has_node(self, node_id: NodeID) -> bool:
        return node_id in self._graph

    async def degree(self, node_id: NodeID) -> int:
        if node_id not in self._graph:
            return 0
        return self._graph.degree(node_id)

    async def get_edges_between(
        self, source: NodeID, target: NodeID, *, relation: str | None = None
    ) -> Sequence[Edge]:
        if not self._graph.has_node(source) or not self._graph.has_node(target):
            return []
        edges: list[Edge] = []
        for _, _, data in self._graph.edges(source, data=True):
            e: Edge = data["edge"]
            if e.target == target and (relation is None or e.relation == relation):
                edges.append(e)
        return edges

    async def list_edges(self, *, relation: str | None = None) -> Sequence[Edge]:
        edges: list[Edge] = []
        for _, _, data in self._graph.edges(data=True):
            e: Edge = data["edge"]
            if relation is None or e.relation == relation:
                edges.append(e)
        return edges

    async def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # --- Graph algorithms ---

    async def bfs(self, start: NodeID, *, relation: str | None = None) -> Sequence[Node[Any]]:
        if start not in self._graph:
            return []
        visited: set[NodeID] = {start}
        queue: deque[NodeID] = deque([start])
        result: list[Node[Any]] = [self._graph.nodes[start]["node"]]
        while queue:
            current = queue.popleft()
            for _, neighbor, _, data in self._graph.out_edges(current, data=True, keys=True):
                if relation is not None and data["edge"].relation != relation:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append(self._graph.nodes[neighbor]["node"])
                    queue.append(neighbor)
        return result

    async def dfs(self, start: NodeID, *, relation: str | None = None) -> Sequence[Node[Any]]:
        if start not in self._graph:
            return []
        visited: set[NodeID] = set()
        result: list[Node[Any]] = []
        stack: list[NodeID] = [start]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            result.append(self._graph.nodes[current]["node"])
            for _, neighbor, _, data in self._graph.out_edges(current, data=True, keys=True):
                if relation is not None and data["edge"].relation != relation:
                    continue
                if neighbor not in visited:
                    stack.append(neighbor)
        return result

    async def shortest_path(
        self, source: NodeID, target: NodeID, *, relation: str | None = None
    ) -> Sequence[Node[Any]] | None:
        if source not in self._graph or target not in self._graph:
            return None
        if relation is not None:
            view = nx.subgraph_view(
                self._graph,
                filter_edge=lambda u, v, k: k == relation,
            )
        else:
            view = self._graph
        try:
            path = nx.shortest_path(view, source, target)
        except nx.NetworkXNoPath:
            return None
        return [self._graph.nodes[nid]["node"] for nid in path]

    async def centrality(self, *, relation: str | None = None) -> dict[NodeID, float]:
        if relation is not None:
            view = nx.subgraph_view(
                self._graph,
                filter_edge=lambda u, v, k: k == relation,
            )
        else:
            view = self._graph
        return nx.degree_centrality(view)

    async def connected_components(
        self, *, relation: str | None = None
    ) -> Sequence[Sequence[NodeID]]:
        if relation is not None:
            view = nx.subgraph_view(
                self._graph,
                filter_edge=lambda u, v, k: k == relation,
            )
        else:
            view = self._graph
        return [list(c) for c in nx.weakly_connected_components(view)]

    async def subgraph(self, node_ids: Sequence[NodeID]) -> NetworkXBackend:
        new = NetworkXBackend()
        sub = self._graph.subgraph(node_ids).copy()
        new._graph = sub
        return new

    async def find_all_paths(
        self, source: NodeID, target: NodeID, *, max_depth: int | None = None
    ) -> Sequence[Sequence[NodeID]]:
        if source not in self._graph or target not in self._graph:
            return []
        cutoff = max_depth if max_depth is not None else len(self._graph)
        return list(nx.all_simple_paths(self._graph, source, target, cutoff=cutoff))

    async def ancestors(self, node_id: NodeID) -> set[NodeID]:
        if node_id not in self._graph:
            return set()
        return nx.ancestors(self._graph, node_id)

    async def descendants(self, node_id: NodeID) -> set[NodeID]:
        if node_id not in self._graph:
            return set()
        return nx.descendants(self._graph, node_id)

    async def ego_graph(self, node_id: NodeID, *, radius: int = 1) -> NetworkXBackend:
        new = NetworkXBackend()
        if node_id not in self._graph:
            return new
        sub = nx.ego_graph(self._graph, node_id, radius=radius)
        new._graph = nx.MultiDiGraph(sub)
        return new

    async def pagerank(
        self, *, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6
    ) -> dict[NodeID, float]:
        nodes = list(self._graph.nodes())
        n = len(nodes)
        if n == 0:
            return {}
        rank = {node: 1.0 / n for node in nodes}
        # Cache out-degrees for efficiency
        out_deg = {node: self._graph.out_degree(node) for node in nodes}
        for _ in range(max_iter):
            # Dangling nodes redistribute rank equally to all nodes
            dangling_sum = sum(rank[node] for node in nodes if out_deg[node] == 0)
            new_rank: dict[NodeID, float] = {}
            for node in nodes:
                incoming = sum(
                    rank[pred] / out_deg[pred]
                    for pred in self._graph.predecessors(node)
                    if out_deg[pred] > 0
                )
                new_rank[node] = (1 - alpha) / n + alpha * (incoming + dangling_sum / n)
            # Check convergence (L1 norm)
            diff = sum(abs(new_rank[node] - rank[node]) for node in nodes)
            rank = new_rank
            if diff < n * tol:
                break
        return rank
