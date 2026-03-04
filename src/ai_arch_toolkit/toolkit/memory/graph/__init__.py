"""Graph storage layer — backends, vector indices, and the GraphStore facade."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.memory.graph._backends import GraphAlgorithms, GraphBackend
from ai_arch_toolkit.toolkit.memory.graph._index import BruteForceIndex, VectorIndex
from ai_arch_toolkit.toolkit.memory.graph._store import GraphStore

# NetworkXBackend is NOT re-exported here (import-guarded).
# Import directly: from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend

__all__ = [
    "BruteForceIndex",
    "GraphAlgorithms",
    "GraphBackend",
    "GraphStore",
    "VectorIndex",
]
