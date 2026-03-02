"""Universal data shapes for the graph-backed memory system."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

type NodeID = str
type NodeType = str
type EmbedFn = Callable[[str], Awaitable[list[float]]]


def _now_utc() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True, kw_only=True)
class Node:
    """A node in the memory graph.

    Content convention: only string values are keyword-searchable.
    """

    id: NodeID = field(default_factory=lambda: uuid.uuid4().hex[:16])
    type: NodeType = "generic"
    content: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
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
class Edge:
    """A directed edge between two nodes."""

    source: NodeID
    target: NodeID
    relation: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchResult:
    """A node with its relevance score."""

    node: Node
    score: float
