"""Knowledge — sync in-memory registry for prompt-injectable reference data."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.knowledge._loaders import (
    load_directory,
    load_json,
    load_markdown,
    load_text,
    load_toml,
    load_yaml,
)
from ai_arch_toolkit.toolkit.knowledge._registry import (
    KnowledgeAlreadyExistsError,
    KnowledgeEntry,
    KnowledgeError,
    KnowledgeRegistry,
)
from ai_arch_toolkit.toolkit.knowledge._search import KnowledgeSearchResult

__all__ = [
    "KnowledgeAlreadyExistsError",
    "KnowledgeEntry",
    "KnowledgeError",
    "KnowledgeRegistry",
    "KnowledgeSearchResult",
    "load_directory",
    "load_json",
    "load_markdown",
    "load_text",
    "load_toml",
    "load_yaml",
]
