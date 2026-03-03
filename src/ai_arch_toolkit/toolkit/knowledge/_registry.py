"""KnowledgeRegistry — sync in-memory registry for prompt-injectable reference data."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import field
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeEntry:
    """A single piece of registered knowledge."""

    key: str
    content: str
    format: str = "text"
    category: str = ""
    tags: frozenset[str] = frozenset()
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""


class KnowledgeRegistry:
    """Sync in-memory registry for prompt-injectable reference data."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[str, KnowledgeEntry] = {}

    # --- Registration ---

    def register(
        self,
        key: str,
        content: str,
        *,
        format: str = "text",
        category: str = "",
        tags: tuple[str, ...] | frozenset[str] = (),
        metadata: dict[str, Any] | None = None,
        source: str = "",
    ) -> KnowledgeEntry:
        """Register knowledge. Silently overwrites if key exists."""
        entry = KnowledgeEntry(
            key=key,
            content=content,
            format=format,
            category=category,
            tags=frozenset(tags),
            metadata=metadata or {},
            source=source,
        )
        self._entries[key] = entry
        return entry

    # --- Retrieval ---

    def get(self, key: str) -> KnowledgeEntry | None:
        return self._entries.get(key)

    def require(self, key: str) -> KnowledgeEntry:
        """Get entry or raise KeyError with available keys."""
        if key not in self._entries:
            available = ", ".join(sorted(self._entries.keys())) or "(none)"
            msg = f"Knowledge key {key!r} not found. Available: {available}"
            raise KeyError(msg)
        return self._entries[key]

    # --- Filtering ---

    def by_category(self, category: str) -> Sequence[KnowledgeEntry]:
        return [e for e in self._entries.values() if e.category == category]

    def by_tags(self, *tags: str, match_all: bool = True) -> Sequence[KnowledgeEntry]:
        tag_set = frozenset(tags)
        if match_all:
            return [e for e in self._entries.values() if tag_set <= e.tags]
        return [e for e in self._entries.values() if tag_set & e.tags]

    # --- Prompt injection ---

    def as_context(
        self,
        *keys: str,
        separator: str = "\n\n---\n\n",
        transform: Callable[[str, str], str] | None = None,
    ) -> str:
        """Build prompt context string from registered knowledge."""
        if not keys:
            return ""
        parts: list[str] = []
        for key in keys:
            entry = self.require(key)
            if transform:
                parts.append(transform(key, entry.content))
            else:
                parts.append(entry.content)
        return separator.join(parts)

    # --- Management ---

    def remove(self, key: str) -> bool:
        if key in self._entries:
            del self._entries[key]
            return True
        return False

    def clear(self) -> None:
        self._entries.clear()

    # --- Introspection ---

    def keys(self) -> list[str]:
        return list(self._entries.keys())

    def categories(self) -> list[str]:
        return sorted({e.category for e in self._entries.values() if e.category})

    def has(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries
