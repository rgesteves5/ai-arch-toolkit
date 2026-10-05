"""Graph storage layer — backends, vector indices, and the GraphStore facade."""

from __future__ import annotations

from typing import TYPE_CHECKING as _TYPE_CHECKING

from ai_arch_toolkit.toolkit.memory.graph._backends import (
    GraphAlgorithms,
    GraphBackend,
    MemoryBackend,
)
from ai_arch_toolkit.toolkit.memory.graph._index import BruteForceIndex, VectorIndex
from ai_arch_toolkit.toolkit.memory.graph._store import GraphStore

if _TYPE_CHECKING:
    # Surfaced lazily (``__getattr__`` below), so networkx, the ``graph`` extra, stays optional.
    from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend

__all__ = [
    "BruteForceIndex",
    "GraphAlgorithms",
    "GraphBackend",
    "GraphStore",
    "MemoryBackend",
    "NetworkXBackend",
    "VectorIndex",
]


def __getattr__(name: str) -> object:
    if name == "NetworkXBackend":
        from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend

        return NetworkXBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
