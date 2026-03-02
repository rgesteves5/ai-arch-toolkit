"""Vector index protocol and brute-force default implementation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ai_arch_toolkit.toolkit.memory._types import NodeID


@runtime_checkable
class VectorIndex(Protocol):
    """Protocol for vector similarity indices."""

    async def add(self, node_id: NodeID, embedding: list[float]) -> None: ...

    async def update(self, node_id: NodeID, embedding: list[float]) -> None: ...

    async def remove(self, node_id: NodeID) -> None: ...

    async def search(
        self, embedding: list[float], *, k: int = 5
    ) -> Sequence[tuple[NodeID, float]]: ...

    async def count(self) -> int: ...


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class BruteForceIndex:
    """In-memory brute-force cosine similarity index.

    Good for up to ~5K nodes. Pure stdlib, no numpy required.
    """

    __slots__ = ("_store",)

    def __init__(self) -> None:
        self._store: dict[NodeID, list[float]] = {}

    async def add(self, node_id: NodeID, embedding: list[float]) -> None:
        self._store[node_id] = embedding

    async def update(self, node_id: NodeID, embedding: list[float]) -> None:
        self._store[node_id] = embedding

    async def remove(self, node_id: NodeID) -> None:
        self._store.pop(node_id, None)

    async def search(
        self, embedding: list[float], *, k: int = 5
    ) -> Sequence[tuple[NodeID, float]]:
        scored = [(nid, _cosine_similarity(embedding, emb)) for nid, emb in self._store.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    async def count(self) -> int:
        return len(self._store)
