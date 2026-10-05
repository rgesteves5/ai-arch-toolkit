"""General-purpose graph layer — Node[T], Edge, Graph, algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING as _TYPE_CHECKING

from ai_arch_toolkit.core.graph._backends import GraphAlgorithms, GraphBackend
from ai_arch_toolkit.core.graph._store import Graph
from ai_arch_toolkit.core.graph._types import Direction, Edge, Node, NodeID, NodeType

if _TYPE_CHECKING:
    # Surfaced lazily (``__getattr__`` below), so networkx, the ``graph`` extra, stays optional.
    from ai_arch_toolkit.core.graph._networkx import NetworkXBackend

__all__ = [
    "Direction",
    "Edge",
    "Graph",
    "GraphAlgorithms",
    "GraphBackend",
    "NetworkXBackend",
    "Node",
    "NodeID",
    "NodeType",
]


def __getattr__(name: str) -> object:
    if name == "NetworkXBackend":
        from ai_arch_toolkit.core.graph._networkx import NetworkXBackend

        return NetworkXBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
