"""General-purpose graph data types."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

type NodeID = str
type NodeType = str
type Direction = Literal["in", "out", "both"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Node[T]:
    """A node in a graph.

    Generic over content type: ``Node[str]``, ``Node[dict]``, ``Node[MyModel]``.
    Content defaults to ``None`` to allow type-only marker nodes.
    """

    id: NodeID = field(default_factory=lambda: uuid.uuid4().hex[:16])
    type: NodeType = "default"
    content: T = field(default=None)  # type: ignore[assignment]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class Edge:
    """A directed edge between two nodes."""

    source: NodeID
    target: NodeID
    relation: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
