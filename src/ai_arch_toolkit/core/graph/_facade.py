"""What both graph facades share: nodes and their type index, edges, bulk ops, the saved shape.

``Graph`` (general purpose) and the memory ``GraphStore`` differ in what a node is and in what
happens around a write (embeddings, access tracking, a vector index), not in how nodes and edges
reach a backend. That part lives here once, generic in the node type, so each facade's backend is
typed by its own nodes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar, Protocol

from ai_arch_toolkit.core._persistence import atomic_write_json
from ai_arch_toolkit.core.graph._backends import GraphAlgorithms
from ai_arch_toolkit.core.graph._types import Direction, Edge, Node, NodeID, NodeType


class NodeStore[N](Protocol):
    """The node, edge and traversal surface a graph backend offers, typed by its nodes."""

    async def add_node(self, node: N) -> None: ...

    async def get_node(self, node_id: NodeID) -> N | None: ...

    async def update_node(self, node_id: NodeID, **attrs: object) -> N | None: ...

    async def remove_node(self, node_id: NodeID) -> bool: ...

    async def list_nodes(
        self, *, type: NodeType | None = None, limit: int | None = None
    ) -> Sequence[N]: ...

    async def count_nodes(self, *, type: NodeType | None = None) -> int: ...

    async def add_edge(self, edge: Edge) -> None: ...

    async def get_edges(
        self, node_id: NodeID, *, direction: Direction = "out", relation: str | None = None
    ) -> Sequence[Edge]: ...

    async def remove_edge(self, source: NodeID, target: NodeID, relation: str) -> bool: ...

    async def neighbors(
        self, node_id: NodeID, *, depth: int = 1, relation: str | None = None
    ) -> Sequence[N]: ...

    async def clear(self, *, type: NodeType | None = None) -> int: ...


class GraphFacade[N: Node[Any]](ABC):
    """Nodes and edges over a backend, with a type index for O(k) ``list``/``count`` by type.

    A subclass says how one of its nodes is saved (``_node_record``) and adds what is its own
    around the writes.
    """

    __slots__ = ("_store", "_type_index")

    _schema_version: ClassVar[int] = 1

    def __init__(self, store: NodeStore[N]) -> None:
        self._store = store
        self._type_index: dict[NodeType, set[NodeID]] = {}

    @property
    def has_algorithms(self) -> bool:
        return isinstance(self._store, GraphAlgorithms)

    # --- Nodes ---

    async def add(self, node: N) -> N:
        """Add a node to the graph."""
        await self._store.add_node(node)
        self._type_index.setdefault(node.type, set()).add(node.id)
        return node

    async def get(self, node_id: NodeID) -> N | None:
        """Get a node by ID."""
        return await self._store.get_node(node_id)

    async def update(self, node_id: NodeID, **attrs: Any) -> N | None:
        """Update node attributes."""
        changed = await self._update(node_id, attrs)
        return changed[1] if changed is not None else None

    async def _update(self, node_id: NodeID, attrs: dict[str, Any]) -> tuple[N, N] | None:
        """The node before and after an update, with the type index kept."""
        old = await self._store.get_node(node_id)
        if old is None:
            return None
        updated = await self._store.update_node(node_id, **attrs)
        if updated is None:
            return None
        if "type" in attrs:
            self._type_index.get(old.type, set()).discard(node_id)
            self._type_index.setdefault(updated.type, set()).add(node_id)
        return old, updated

    async def remove(self, node_id: NodeID) -> bool:
        """Remove a node."""
        node = await self._store.get_node(node_id)
        if node is None:
            return False
        removed = await self._store.remove_node(node_id)
        if removed:
            self._type_index.get(node.type, set()).discard(node_id)
        return removed

    async def list(self, *, type: NodeType | None = None, limit: int | None = None) -> Sequence[N]:
        """List nodes, using the type index for O(k) lookup when available."""
        if type is None or type not in self._type_index:
            return await self._store.list_nodes(type=type, limit=limit)
        nodes: list[N] = []
        for node_id in self._type_index[type]:
            node = await self._store.get_node(node_id)
            if node is not None:
                nodes.append(node)
                if limit is not None and len(nodes) >= limit:
                    break
        return nodes

    async def count(self, *, type: NodeType | None = None) -> int:
        """Count nodes, using the type index when available."""
        if type is not None and type in self._type_index:
            return len(self._type_index[type])
        return await self._store.count_nodes(type=type)

    async def clear(self, *, type: NodeType | None = None) -> int:
        """Clear nodes (of one type, or all)."""
        count = await self._store.clear(type=type)
        if type is None:
            self._type_index.clear()
        else:
            self._type_index.pop(type, None)
        return count

    async def add_many(self, nodes: Sequence[N]) -> int:
        """Add multiple nodes. Returns count added."""
        for node in nodes:
            await self.add(node)
        return len(nodes)

    async def remove_many(self, node_ids: Sequence[NodeID]) -> int:
        """Remove multiple nodes. Returns count removed."""
        return sum([await self.remove(node_id) for node_id in node_ids])

    def _indexed_ids(self, type: NodeType | None) -> list[NodeID]:
        if type is None:
            return [node_id for ids in self._type_index.values() for node_id in ids]
        return list(self._type_index.get(type, set()))

    # --- Edges ---

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
        await self._store.add_edge(edge)
        return edge

    async def edges(
        self, node_id: NodeID, *, direction: Direction = "out", relation: str | None = None
    ) -> Sequence[Edge]:
        return await self._store.get_edges(node_id, direction=direction, relation=relation)

    async def disconnect(self, source: NodeID, target: NodeID, relation: str) -> bool:
        return await self._store.remove_edge(source, target, relation)

    async def neighbors(
        self, node_id: NodeID, *, depth: int = 1, relation: str | None = None
    ) -> Sequence[N]:
        return await self._store.neighbors(node_id, depth=depth, relation=relation)

    # --- Persistence ---

    async def to_dict(self) -> dict[str, Any]:
        """Serialize all nodes and edges to a dict."""
        nodes = await self._store.list_nodes()
        edges: list[dict[str, Any]] = []
        for node in nodes:
            for edge in await self._store.get_edges(node.id, direction="out"):
                edges.append(
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "relation": edge.relation,
                        "weight": edge.weight,
                        "metadata": edge.metadata,
                    }
                )
        return {
            "schema_version": self._schema_version,
            "nodes": [self._node_record(node) for node in nodes],
            "edges": edges,
        }

    async def save(self, path: str | Path) -> None:
        """Save the graph to a JSON file."""
        atomic_write_json(path, await self.to_dict())

    @abstractmethod
    def _node_record(self, node: N) -> dict[str, Any]:
        """One node as a saved graph holds it."""


def edges_from(records: Sequence[dict[str, Any]]) -> list[Edge]:
    """The edges of a saved graph (checked by ``check_payload``)."""
    return [
        Edge(
            source=record["source"],
            target=record["target"],
            relation=record["relation"],
            weight=record.get("weight", 1.0),
            metadata=record.get("metadata", {}),
        )
        for record in records
    ]


def check_payload(
    data: dict[str, Any],
    label: str,
    max_version: int,
    check_node: Callable[[int, dict[str, Any]], None],
) -> None:
    """Check a saved graph: its version, its node records (each with ``check_node``), its edges.

    Raises:
        ValueError: The first thing wrong, named with ``label`` ("Graph", "Memory graph").
    """
    version = data.get("schema_version", 0)
    if not isinstance(version, int):
        raise ValueError(f"{label} payload schema_version must be an integer")
    if version > max_version:
        msg = f"Unsupported {label.lower()} schema_version {version}; "
        raise ValueError(msg + f"maximum supported is {max_version}")
    if version < 0:
        raise ValueError(f"Unsupported {label.lower()} schema_version {version}")
    nodes, edges = data.get("nodes"), data.get("edges")
    if not isinstance(nodes, list):
        raise ValueError(f"{label} payload must contain a 'nodes' list")
    if not isinstance(edges, list):
        raise ValueError(f"{label} payload must contain an 'edges' list")
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"{label} node at index {index} must be an object")
        check_node(index, node)
    for index, edge in enumerate(edges):
        _check_edge(edge, index, label)


def _check_edge(edge: object, index: int, label: str) -> None:
    if not isinstance(edge, dict):
        raise ValueError(f"{label} edge at index {index} must be an object")
    for field in ("source", "target", "relation"):
        if field not in edge:
            raise ValueError(f"{label} edge at index {index} missing required field {field!r}")
    if "metadata" in edge and not isinstance(edge["metadata"], dict):
        raise ValueError(f"{label} edge at index {index} metadata must be an object")
