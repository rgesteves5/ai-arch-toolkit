"""Graph backend and algorithm protocols — re-exported from core."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ai_arch_toolkit.core.graph import (  # noqa: F401
    GraphAlgorithms,
    GraphBackend,
)
from ai_arch_toolkit.toolkit.memory._types import Node, NodeType


@runtime_checkable
class MemoryBackend(Protocol):
    """Extended backend protocol for memory-specific search operations.

    Backends used by GraphStore must implement both ``GraphBackend`` (from core)
    and these additional search methods.
    """

    async def search_content(
        self, query: str, *, type: NodeType | None = None, k: int = 5
    ) -> Sequence[Node]: ...

    async def search_similar(
        self, embedding: list[float], *, type: NodeType | None = None, k: int = 5
    ) -> Sequence[Node] | None: ...
