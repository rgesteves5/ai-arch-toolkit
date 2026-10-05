"""Graph-backed memory system for LLM agents."""

from __future__ import annotations

from typing import TYPE_CHECKING as _TYPE_CHECKING

from ai_arch_toolkit.toolkit.memory._middleware import MemoryMiddleware
from ai_arch_toolkit.toolkit.memory._presets import MemoryPreset, cognitive, conversational
from ai_arch_toolkit.toolkit.memory._tools import memory_tools
from ai_arch_toolkit.toolkit.memory._types import Edge, Node, NodeID, NodeType, SearchResult
from ai_arch_toolkit.toolkit.memory._views import (
    PropertyView,
    RelationalView,
    SimilarityView,
    TemporalView,
    composite_score,
)
from ai_arch_toolkit.toolkit.memory.graph import (
    BruteForceIndex,
    GraphAlgorithms,
    GraphBackend,
    GraphStore,
    VectorIndex,
)

if _TYPE_CHECKING:
    # Surfaced lazily (``__getattr__`` below), so networkx, the ``graph`` extra, stays optional.
    from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend

__all__ = [
    "BruteForceIndex",
    "Edge",
    "GraphAlgorithms",
    "GraphBackend",
    "GraphStore",
    "MemoryMiddleware",
    "MemoryPreset",
    "NetworkXBackend",
    "Node",
    "NodeID",
    "NodeType",
    "PropertyView",
    "RelationalView",
    "SearchResult",
    "SimilarityView",
    "TemporalView",
    "VectorIndex",
    "cognitive",
    "composite_score",
    "conversational",
    "memory_tools",
]


def __getattr__(name: str) -> object:
    if name == "NetworkXBackend":
        from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend

        return NetworkXBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
