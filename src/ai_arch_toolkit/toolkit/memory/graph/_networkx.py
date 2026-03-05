"""NetworkX-based memory graph backend — extends core with search methods."""

from __future__ import annotations

from collections.abc import Sequence

from ai_arch_toolkit.core.graph._networkx import NetworkXBackend as _CoreNetworkXBackend
from ai_arch_toolkit.toolkit.memory._types import Node, NodeType


def _keyword_score(node: Node, tokens: list[str]) -> float:
    """Score a node by token overlap with string values in content."""
    searchable = " ".join(str(v).lower() for v in node.content.values() if isinstance(v, str))
    if not searchable:
        return 0.0
    hits = sum(1 for t in tokens if t in searchable)
    return hits / len(tokens) if tokens else 0.0


class NetworkXBackend(_CoreNetworkXBackend):
    """Memory-compatible backend. Adds search methods needed by GraphStore."""

    async def search_content(
        self, query: str, *, type: NodeType | None = None, k: int = 5
    ) -> Sequence[Node]:
        tokens = query.lower().split()
        if not tokens:
            return []
        scored: list[tuple[Node, float]] = []
        for _, data in self._graph.nodes(data=True):
            node: Node = data["node"]
            if type is not None and node.type != type:
                continue
            score = _keyword_score(node, tokens)
            if score > 0:
                scored.append((node, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in scored[:k]]

    async def search_similar(
        self, embedding: list[float], *, type: NodeType | None = None, k: int = 5
    ) -> Sequence[Node] | None:
        return None
