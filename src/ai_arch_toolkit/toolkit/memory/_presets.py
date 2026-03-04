"""Pre-configured view bundles for common memory patterns."""

from __future__ import annotations

from collections.abc import KeysView
from dataclasses import dataclass, field
from typing import Any

from ai_arch_toolkit.toolkit.memory._views import (
    PropertyView,
    RelationalView,
    SimilarityView,
    TemporalView,
)
from ai_arch_toolkit.toolkit.memory.graph._store import GraphStore

type _AnyView = TemporalView | SimilarityView | RelationalView | PropertyView


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryPreset:
    """A named bundle of views over a GraphStore."""

    store: GraphStore
    views: dict[str, _AnyView] = field(default_factory=dict)

    def __getitem__(self, key: str) -> _AnyView:
        return self.views[key]

    def keys(self) -> KeysView[str]:
        return self.views.keys()

    async def consolidate(self) -> int:
        """Remove duplicate nodes by content key. Returns count removed."""
        all_nodes = await self.store.list()
        seen: dict[str, str] = {}  # content_key → node_id
        removed = 0
        for node in all_nodes:
            key = _content_key(node.content)
            if key in seen:
                await self.store.remove(node.id)
                removed += 1
            else:
                seen[key] = node.id
        return removed


def _content_key(content: dict[str, Any]) -> str:
    """Create a hashable key from content dict for dedup."""
    parts = sorted(
        (k, str(v)) for k, v in content.items() if isinstance(v, (str, int, float, bool))
    )
    return "|".join(f"{k}={v}" for k, v in parts)


def cognitive(store: GraphStore) -> MemoryPreset:
    """Human-inspired memory preset: semantic, episodic, procedural.

    Views:
        - semantic: SimilarityView for facts
        - episodic: TemporalView for events
        - procedural: SimilarityView for rules
        - relations: RelationalView (all types)
        - properties: PropertyView (all types)
    """
    return MemoryPreset(
        store=store,
        views={
            "semantic": SimilarityView(store, node_type="fact"),
            "episodic": TemporalView(store, node_type="event"),
            "procedural": SimilarityView(store, node_type="rule"),
            "relations": RelationalView(store),
            "properties": PropertyView(store),
        },
    )


def conversational(store: GraphStore) -> MemoryPreset:
    """Conversational memory preset: history, preferences, knowledge.

    Views:
        - history: TemporalView for interactions
        - preferences: PropertyView for preferences
        - knowledge: SimilarityView for facts
    """
    return MemoryPreset(
        store=store,
        views={
            "history": TemporalView(store, node_type="interaction"),
            "preferences": PropertyView(store, node_type="preference"),
            "knowledge": SimilarityView(store, node_type="fact"),
        },
    )
