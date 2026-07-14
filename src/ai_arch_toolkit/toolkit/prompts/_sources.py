"""Resolved and dynamic sources for prompt-template sections."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from ai_arch_toolkit.toolkit.prompts._template_engines import StringTemplateEngine
from ai_arch_toolkit.toolkit.resources import (
    Resource,
    ResourcePolicy,
    ResourceResolver,
    ResourceSelector,
    ResourceSerializer,
    load_resource,
    select_resource,
    serialize_resource_value,
)

if TYPE_CHECKING:
    from ai_arch_toolkit.toolkit.knowledge import KnowledgeRegistry


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceResolution:
    """Literal section content plus non-sensitive source provenance."""

    content: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("SourceResolution.content must be a string")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("SourceResolution.provenance must be a mapping")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


class PromptSource(Protocol):
    """Resolve prompt-section content without performing model calls."""

    def resolve(self, variables: Mapping[str, Any]) -> SourceResolution:
        """Resolve content using validated prompt variables."""
        ...

    def describe(self) -> Mapping[str, Any]:
        """Return non-sensitive provenance without resolving dynamic content."""
        ...


@dataclass(frozen=True, slots=True)
class LiteralSource:
    """Return literal inline content."""

    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("LiteralSource.content must be a string")

    def resolve(self, variables: Mapping[str, Any]) -> SourceResolution:
        return SourceResolution(content=self.content, provenance=self.describe())

    def describe(self) -> Mapping[str, Any]:
        import hashlib

        digest = hashlib.sha256(self.content.encode()).hexdigest()
        return {"kind": "literal", "content_fingerprint": f"sha256:{digest}"}


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceSource:
    """Select and serialize an already loaded resource snapshot."""

    resource: Resource
    selector: str | ResourceSelector | None = None
    serialize_as: str | ResourceSerializer | None = None
    template_selector: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.resource, Resource):
            raise TypeError("ResourceSource.resource must be a Resource")
        if (
            self.selector is not None
            and not isinstance(self.selector, str)
            and not hasattr(self.selector, "select")
        ):
            raise TypeError("ResourceSource.selector must be a string, ResourceSelector, or None")
        if (
            self.serialize_as is not None
            and not isinstance(self.serialize_as, str)
            and not hasattr(self.serialize_as, "serialize")
        ):
            raise TypeError("ResourceSource.serialize_as must be a name, serializer, or None")
        if not isinstance(self.template_selector, bool):
            raise TypeError("ResourceSource.template_selector must be a boolean")

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        selector: str | ResourceSelector | None = None,
        serialize_as: str | ResourceSerializer | None = None,
        policy: ResourcePolicy | None = None,
        resolver: ResourceResolver | None = None,
    ) -> ResourceSource:
        """Eagerly load a file into a reusable resource source."""
        serializer = (
            resolver.serializers.resolve(serialize_as)
            if resolver is not None and serialize_as is not None
            else serialize_as
        )
        return cls(
            resource=load_resource(path, policy=policy, resolver=resolver),
            selector=selector,
            serialize_as=serializer,
        )

    def resolve(self, variables: Mapping[str, Any]) -> SourceResolution:
        selector = self.selector
        if self.template_selector and isinstance(selector, str) and "$" in selector:
            selector = StringTemplateEngine().render(selector, variables)
        if selector is None and self.serialize_as is None and self.resource.text is not None:
            content = self.resource.text
        else:
            value = select_resource(self.resource, selector)
            content = serialize_resource_value(value, as_format=self.serialize_as or "text")
        provenance = {
            **self.describe(),
            "selector": selector if isinstance(selector, str) else _selector_name(selector),
        }
        return SourceResolution(content=content, provenance=provenance)

    def describe(self) -> Mapping[str, Any]:
        return {
            "kind": "resource",
            "source": self.resource.ref.uri,
            "resolved_source": (
                self.resource.provenance.resolved_uri if self.resource.provenance else None
            ),
            "media_type": self.resource.media_type,
            "resource_fingerprint": self.resource.fingerprint,
            "selector": (
                self.selector if isinstance(self.selector, str) else _selector_name(self.selector)
            ),
            "serialize_as": (
                self.serialize_as
                if isinstance(self.serialize_as, str | type(None))
                else getattr(self.serialize_as, "name", type(self.serialize_as).__name__)
            ),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeSource:
    """Resolve registered knowledge entries into one prompt section."""

    registry: KnowledgeRegistry = field(compare=False, hash=False, repr=False)
    keys: tuple[str, ...]
    separator: str = "\n\n---\n\n"
    include_names: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", tuple(self.keys))
        if not all(isinstance(key, str) and key for key in self.keys):
            raise ValueError("KnowledgeSource.keys must contain non-empty strings")
        if not isinstance(self.separator, str):
            raise TypeError("KnowledgeSource.separator must be a string")
        if not isinstance(self.include_names, bool):
            raise TypeError("KnowledgeSource.include_names must be a boolean")

    def resolve(self, variables: Mapping[str, Any]) -> SourceResolution:
        entries = [self.registry.require(key) for key in self.keys]
        if self.include_names:
            parts = [f"[{entry.key}]\n{entry.content}" for entry in entries]
        else:
            parts = [entry.content for entry in entries]
        return SourceResolution(
            content=self.separator.join(parts),
            provenance={**self.describe(), "sources": tuple(entry.source for entry in entries)},
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "kind": "knowledge",
            "keys": self.keys,
            "separator": self.separator,
            "include_names": self.include_names,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class CallableSource:
    """Python-only source for application-controlled dynamic content."""

    function: Callable[[Mapping[str, Any]], str] = field(compare=False, hash=False, repr=False)
    name: str = "callable"

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("CallableSource.function must be callable")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("CallableSource.name is required")

    def resolve(self, variables: Mapping[str, Any]) -> SourceResolution:
        content = self.function(variables)
        if not isinstance(content, str):
            raise TypeError(f"prompt callable source {self.name!r} must return a string")
        return SourceResolution(
            content=content,
            provenance=self.describe(),
        )

    def describe(self) -> Mapping[str, Any]:
        return {"kind": "callable", "name": self.name}


def _selector_name(selector: ResourceSelector | None) -> str | None:
    return None if selector is None else type(selector).__name__


def knowledge_source(
    registry: KnowledgeRegistry,
    keys: Sequence[str],
    *,
    separator: str = "\n\n---\n\n",
    include_names: bool = False,
) -> KnowledgeSource:
    """Create a knowledge source from any finite key sequence."""
    return KnowledgeSource(
        registry=registry,
        keys=tuple(keys),
        separator=separator,
        include_names=include_names,
    )


__all__ = [
    "CallableSource",
    "KnowledgeSource",
    "LiteralSource",
    "PromptSource",
    "ResourceSource",
    "SourceResolution",
    "knowledge_source",
]
