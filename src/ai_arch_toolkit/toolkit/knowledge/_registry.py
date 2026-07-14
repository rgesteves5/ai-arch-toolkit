"""KnowledgeRegistry — sync in-memory registry for prompt-injectable reference data."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ai_arch_toolkit.toolkit.resources import (
    Resource,
    ResourcePolicy,
    ResourceResolver,
    ResourceSelector,
    ResourceSerializer,
    SerializerRegistry,
    load_resource,
    load_resources,
    select_resource,
    serialize_resource_value,
)

if TYPE_CHECKING:
    from ai_arch_toolkit.toolkit.knowledge._search import KnowledgeSearchResult


class KnowledgeError(Exception):
    """Base class for knowledge registry failures."""


class KnowledgeAlreadyExistsError(KnowledgeError, ValueError):
    """A knowledge key already exists and overwrite was not requested."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeEntry:
    """A single piece of registered knowledge."""

    key: str
    content: str
    format: str = "text"
    category: str = ""
    tags: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)
    source: str = ""
    resource: Resource | None = field(default=None, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("KnowledgeEntry.key is required")
        if not isinstance(self.content, str):
            raise TypeError("KnowledgeEntry.content must be a string")
        for field_name, value in (
            ("format", self.format),
            ("category", self.category),
            ("source", self.source),
        ):
            if not isinstance(value, str):
                raise TypeError(f"KnowledgeEntry.{field_name} must be a string")
        try:
            normalized_tags = frozenset(self.tags) if not isinstance(self.tags, str) else None
        except TypeError:
            normalized_tags = None
        if normalized_tags is None or not all(isinstance(tag, str) for tag in normalized_tags):
            raise TypeError("KnowledgeEntry.tags must contain strings")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("KnowledgeEntry.metadata must be a mapping")
        if self.resource is not None and not isinstance(self.resource, Resource):
            raise TypeError("KnowledgeEntry.resource must be a Resource or None")
        object.__setattr__(self, "tags", normalized_tags)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def data(self) -> Any:
        """Return parsed resource data when available, otherwise literal content."""
        return self.resource.data if self.resource is not None else self.content

    @property
    def fingerprint(self) -> str:
        """Return the source-resource or literal-content fingerprint."""
        if self.resource is not None:
            return self.resource.fingerprint
        return "sha256:" + hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def media_type(self) -> str:
        """Return a media type for the entry's content."""
        if self.resource is not None:
            return self.resource.media_type
        return {
            "json": "application/json",
            "markdown": "text/markdown",
            "toml": "application/toml",
            "yaml": "application/yaml",
        }.get(self.format, "text/plain")


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
        resource: Resource | None = None,
        overwrite: bool = False,
    ) -> KnowledgeEntry:
        """Register knowledge, requiring explicit overwrite on key conflicts."""
        if key in self._entries and not overwrite:
            raise KnowledgeAlreadyExistsError(
                f"knowledge key {key!r} already exists; pass overwrite=True to replace it"
            )
        entry = KnowledgeEntry(
            key=key,
            content=content,
            format=format,
            category=category,
            tags=frozenset(tags),
            metadata=metadata or {},
            source=source,
            resource=resource,
        )
        self._entries[key] = entry
        return entry

    def register_resource(
        self,
        key: str,
        resource: Resource,
        *,
        selector: str | ResourceSelector | None = None,
        serialize_as: str | ResourceSerializer | None = None,
        serializer_registry: SerializerRegistry | None = None,
        category: str = "",
        tags: tuple[str, ...] | frozenset[str] = (),
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> KnowledgeEntry:
        """Select, serialize, and register a loaded resource."""
        if not isinstance(resource, Resource):
            raise TypeError("resource must be a Resource")
        if selector is None and serialize_as is None and resource.text is not None:
            content = resource.text
        else:
            value = select_resource(resource, selector)
            content = serialize_resource_value(
                value,
                as_format=serialize_as or "text",
                registry=serializer_registry,
            )
        format_name = (
            serialize_as if isinstance(serialize_as, str) else getattr(serialize_as, "name", None)
        )
        if not isinstance(format_name, str) or not format_name:
            format_name = _format_from_media_type(resource.media_type)
        return self.register(
            key,
            content,
            format=format_name,
            category=category,
            tags=tags,
            metadata=metadata,
            source=resource.ref.uri,
            resource=resource,
            overwrite=overwrite,
        )

    def load(
        self,
        key: str,
        path: str | Path,
        *,
        selector: str | ResourceSelector | None = None,
        serialize_as: str | None = None,
        category: str = "",
        tags: tuple[str, ...] | frozenset[str] = (),
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
        policy: ResourcePolicy | None = None,
        resolver: ResourceResolver | None = None,
    ) -> KnowledgeEntry:
        """Load and register one file through the resource subsystem."""
        resource = load_resource(path, policy=policy, resolver=resolver)
        return self.register_resource(
            key,
            resource,
            selector=selector,
            serialize_as=serialize_as,
            category=category,
            tags=tags,
            metadata=metadata,
            serializer_registry=resolver.serializers if resolver is not None else None,
            overwrite=overwrite,
        )

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        pattern: str = "*",
        recursive: bool = False,
        prefix: str = "",
        extensions: set[str] | None = None,
        category: str = "",
        tags: tuple[str, ...] = (),
        policy: ResourcePolicy | None = None,
        resolver: ResourceResolver | None = None,
    ) -> KnowledgeRegistry:
        """Create a registry from known resources in one directory."""
        registry = cls()
        root = Path(directory).expanduser().resolve()
        resources = load_resources(
            root,
            pattern=pattern,
            recursive=recursive,
            extensions=extensions,
            policy=policy,
            resolver=resolver,
        )
        for resource in resources:
            source_path = Path(resource.provenance.resolved_uri) if resource.provenance else None
            if source_path is None:
                raise ValueError("directory resources must provide a resolved filesystem path")
            relative = source_path.relative_to(root)
            key_path = relative.with_suffix("") if recursive else Path(relative.stem)
            key = prefix + key_path.as_posix().replace("/", ".")
            if registry.has(key):
                raise ValueError(f"Stem collision while loading directory: {key!r}")
            registry.register_resource(
                key,
                resource,
                category=category,
                tags=tags,
                serializer_registry=resolver.serializers if resolver is not None else None,
            )
        return registry

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

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        category: str | None = None,
        tags: tuple[str, ...] = (),
        match_all_tags: bool = True,
    ) -> tuple[KnowledgeSearchResult, ...]:
        """Search entries lexically with deterministic, explainable relevance."""
        from ai_arch_toolkit.toolkit.knowledge._search import search_entries

        entries = tuple(self._entries.values())
        if category is not None:
            entries = tuple(entry for entry in entries if entry.category == category)
        if tags:
            required = frozenset(tags)
            entries = tuple(
                entry
                for entry in entries
                if (required <= entry.tags if match_all_tags else bool(required & entry.tags))
            )
        return search_entries(entries, query, limit=limit)

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


def _format_from_media_type(media_type: str) -> str:
    return {
        "application/json": "json",
        "application/toml": "toml",
        "application/x-yaml": "yaml",
        "application/yaml": "yaml",
        "text/markdown": "markdown",
        "text/yaml": "yaml",
    }.get(media_type, "text")
