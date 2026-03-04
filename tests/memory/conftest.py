"""Shared fixtures for memory tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from ai_arch_toolkit.toolkit.memory._types import Node
from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend
from ai_arch_toolkit.toolkit.memory.graph._store import GraphStore


def make_node(
    *,
    id: str = "",
    type: str = "generic",
    content: dict[str, Any] | None = None,
    source: str = "unknown",
    confidence: float = 1.0,
    **kwargs: Any,
) -> Node:
    """Factory for test nodes."""
    kw: dict[str, Any] = {
        "type": type,
        "content": content or {},
        "source": source,
        "confidence": confidence,
        **kwargs,
    }
    if id:
        kw["id"] = id
    return Node(**kw)


def mock_embed_fn() -> AsyncMock:
    """Create a mock embed function that returns deterministic embeddings."""

    async def _embed(text: str) -> list[float]:
        # Simple deterministic embedding: hash-based 4-dim vector
        h = hash(text) % 10000
        return [
            (h % 100) / 100.0,
            ((h // 100) % 100) / 100.0,
            ((h // 10000) % 100) / 100.0,
            (h % 50) / 50.0,
        ]

    return AsyncMock(side_effect=_embed)


@pytest.fixture
def backend() -> NetworkXBackend:
    return NetworkXBackend()


@pytest.fixture
def store(backend: NetworkXBackend) -> GraphStore:
    return GraphStore(backend)


@pytest.fixture
def store_with_embed(backend: NetworkXBackend) -> tuple[GraphStore, AsyncMock]:
    embed = mock_embed_fn()
    s = GraphStore(backend, embed=embed)
    return s, embed
