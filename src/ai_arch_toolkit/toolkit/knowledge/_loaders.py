"""Compatibility file loaders delegating to ``toolkit.resources``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_arch_toolkit.toolkit.knowledge._registry import KnowledgeEntry, KnowledgeRegistry
from ai_arch_toolkit.toolkit.resources import ResourceDecodeError


def load_text(
    registry: KnowledgeRegistry,
    key: str,
    path: str | Path,
    **kw: Any,
) -> KnowledgeEntry:
    """Load a plain-text resource into a knowledge registry."""
    return registry.load(key, path, serialize_as=None, overwrite=True, **kw)


def load_json(
    registry: KnowledgeRegistry,
    key: str,
    path: str | Path,
    **kw: Any,
) -> KnowledgeEntry:
    """Load JSON and preserve the legacy indented JSON entry content."""
    try:
        return registry.load(key, path, serialize_as="json", overwrite=True, **kw)
    except ResourceDecodeError as exc:
        if isinstance(exc.__cause__, json.JSONDecodeError):
            raise exc.__cause__ from exc
        raise


def load_toml(
    registry: KnowledgeRegistry,
    key: str,
    path: str | Path,
    **kw: Any,
) -> KnowledgeEntry:
    """Load and validate TOML while preserving its source text."""
    return registry.load(key, path, serialize_as=None, overwrite=True, **kw)


def load_markdown(
    registry: KnowledgeRegistry,
    key: str,
    path: str | Path,
    **kw: Any,
) -> KnowledgeEntry:
    """Load a Markdown resource into a knowledge registry."""
    return registry.load(key, path, serialize_as=None, overwrite=True, **kw)


def load_yaml(
    registry: KnowledgeRegistry,
    key: str,
    path: str | Path,
    **kw: Any,
) -> KnowledgeEntry:
    """Load and validate YAML while preserving its source text."""
    return registry.load(key, path, serialize_as=None, overwrite=True, **kw)


def load_directory(
    registry: KnowledgeRegistry,
    directory: str | Path,
    *,
    pattern: str = "*",
    recursive: bool = False,
    prefix: str = "",
    extensions: set[str] | None = None,
    category: str = "",
    tags: tuple[str, ...] = (),
) -> int:
    """Load known directory resources while preserving legacy key generation."""
    loaded = KnowledgeRegistry.from_directory(
        directory,
        pattern=pattern,
        recursive=recursive,
        prefix=prefix,
        extensions=extensions,
        category=category,
        tags=tags,
    )
    for key in loaded.keys():  # noqa: SIM118 - KnowledgeRegistry is not a mapping
        entry = loaded.require(key)
        registry.register(
            entry.key,
            entry.content,
            format=entry.format,
            category=entry.category,
            tags=entry.tags,
            metadata=dict(entry.metadata),
            source=entry.source,
            resource=entry.resource,
            overwrite=True,
        )
    return len(loaded)


__all__ = [
    "load_directory",
    "load_json",
    "load_markdown",
    "load_text",
    "load_toml",
    "load_yaml",
]
