"""Graph-backed memory system for LLM agents."""

from __future__ import annotations

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

__all__ = [
    "BruteForceIndex",
    "Edge",
    "GraphAlgorithms",
    "GraphBackend",
    "GraphStore",
    "MemoryMiddleware",
    "MemoryPreset",
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
