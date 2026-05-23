"""Universal data shapes for the graph-backed memory system."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ai_arch_toolkit.core.graph import Edge, NodeID, NodeType  # noqa: F401
from ai_arch_toolkit.core.graph._types import Node as _GraphNode

type EmbedFn = Callable[[str], Awaitable[list[float]]]


def _now_utc() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True, kw_only=True)
class Node(_GraphNode[dict[str, Any]]):
    """A node in the memory graph — a ``core.graph.Node[dict[str, Any]]`` with
    the bookkeeping memory needs.

    Inheriting from the core graph ``Node`` means a ``MemoryBackend`` IS a
    ``GraphBackend[dict[str, Any]]`` for free — no parallel type hierarchy,
    no runtime-only duck typing. Content convention: only string values are
    keyword-searchable.
    """

    type: NodeType = "generic"
    content: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    # Temporal (bi-temporal)
    timestamp: datetime = field(default_factory=_now_utc)
    created_at: datetime = field(default_factory=_now_utc)
    # Lifecycle
    access_count: int = 0
    last_accessed: datetime | None = None
    confidence: float = 1.0
    # Provenance
    source: str = "unknown"


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchResult:
    """A node with its relevance score."""

    node: Node
    score: float
