"""Graph backend and algorithm protocols — re-exported from core."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ai_arch_toolkit.core.graph import (  # noqa: F401
    GraphAlgorithms,
    GraphBackend,
)
from ai_arch_toolkit.core.graph._types import Direction, Edge
from ai_arch_toolkit.toolkit.memory._types import Node, NodeID, NodeType


@runtime_checkable
class MemoryBackend(Protocol):
    """Backend protocol for the memory graph.

    Combines the node / edge CRUD surface of ``core.graph.GraphBackend`` with
    the memory-specific keyword + vector search methods. Methods inherited
    from ``GraphBackend`` are re-declared here with the narrower memory
    ``Node`` return type (memory ``Node`` is a subclass of
    ``core.graph.Node[dict[str, Any]]``, so the override is covariant).
    """

    # --- Node operations (narrowed from GraphBackend to memory ``Node``) ---

    async def add_node(self, node: Node) -> None: ...

    async def get_node(self, node_id: NodeID) -> Node | None: ...

    async def update_node(self, node_id: NodeID, **attrs: object) -> Node | None: ...

    async def remove_node(self, node_id: NodeID) -> bool: ...

    async def list_nodes(
        self, *, type: NodeType | None = None, limit: int | None = None
    ) -> Sequence[Node]: ...

    async def count_nodes(self, *, type: NodeType | None = None) -> int: ...

    # --- Edge operations (unchanged from GraphBackend) ---

    async def add_edge(self, edge: Edge) -> None: ...

    async def get_edges(
        self,
        node_id: NodeID,
        *,
        direction: Direction = "out",
        relation: str | None = None,
    ) -> Sequence[Edge]: ...

    async def remove_edge(self, source: NodeID, target: NodeID, relation: str) -> bool: ...

    async def neighbors(
        self,
        node_id: NodeID,
        *,
        depth: int = 1,
        relation: str | None = None,
    ) -> Sequence[Node]: ...

    # --- Bulk ---

    async def clear(self, *, type: NodeType | None = None) -> int: ...

    # --- Memory-specific search ---

    async def search_content(
        self, query: str, *, type: NodeType | None = None, k: int = 5
    ) -> Sequence[Node]: ...

    async def search_similar(
        self, embedding: list[float], *, type: NodeType | None = None, k: int = 5
    ) -> Sequence[Node] | None: ...
