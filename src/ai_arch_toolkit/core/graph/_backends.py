"""Graph backend and algorithm protocols."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from ai_arch_toolkit.core.graph._types import Direction, Edge, Node, NodeID, NodeType


@runtime_checkable
class GraphBackend(Protocol):
    """Protocol for graph storage backends.

    All methods are async. Implementations must structurally match this protocol.
    """

    # --- Node operations ---

    async def add_node(self, node: Node[Any]) -> None: ...

    async def get_node(self, node_id: NodeID) -> Node[Any] | None: ...

    async def update_node(self, node_id: NodeID, **attrs: object) -> Node[Any] | None: ...

    async def remove_node(self, node_id: NodeID) -> bool: ...

    async def list_nodes(
        self, *, type: NodeType | None = None, limit: int | None = None
    ) -> Sequence[Node[Any]]: ...

    async def count_nodes(self, *, type: NodeType | None = None) -> int: ...

    # --- Edge operations ---

    async def add_edge(self, edge: Edge) -> None: ...

    async def get_edges(
        self,
        node_id: NodeID,
        *,
        direction: Direction = "out",
        relation: str | None = None,
    ) -> Sequence[Edge]: ...

    async def remove_edge(self, source: NodeID, target: NodeID, relation: str) -> bool: ...

    # --- Traversal ---

    async def neighbors(
        self, node_id: NodeID, *, depth: int = 1, relation: str | None = None
    ) -> Sequence[Node[Any]]: ...

    # --- Bulk ---

    async def clear(self, *, type: NodeType | None = None) -> int: ...


@runtime_checkable
class GraphAlgorithms(Protocol):
    """Optional protocol for graph algorithms (BFS, DFS, etc.)."""

    async def bfs(self, start: NodeID, *, relation: str | None = None) -> Sequence[Node[Any]]: ...

    async def dfs(self, start: NodeID, *, relation: str | None = None) -> Sequence[Node[Any]]: ...

    async def shortest_path(
        self, source: NodeID, target: NodeID, *, relation: str | None = None
    ) -> Sequence[Node[Any]] | None: ...

    async def centrality(self, *, relation: str | None = None) -> dict[NodeID, float]: ...

    async def connected_components(
        self, *, relation: str | None = None
    ) -> Sequence[Sequence[NodeID]]: ...

    async def subgraph(self, node_ids: Sequence[NodeID]) -> GraphBackend: ...
