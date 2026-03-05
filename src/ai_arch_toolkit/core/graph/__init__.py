"""General-purpose graph layer — Node[T], Edge, Graph, algorithms."""

from __future__ import annotations

from ai_arch_toolkit.core.graph._backends import GraphAlgorithms, GraphBackend
from ai_arch_toolkit.core.graph._store import Graph
from ai_arch_toolkit.core.graph._types import Direction, Edge, Node, NodeID, NodeType

# NetworkXBackend is NOT re-exported here (import-guarded).
# Import directly: from ai_arch_toolkit.core.graph._networkx import NetworkXBackend

__all__ = [
    "Direction",
    "Edge",
    "Graph",
    "GraphAlgorithms",
    "GraphBackend",
    "Node",
    "NodeID",
    "NodeType",
]
