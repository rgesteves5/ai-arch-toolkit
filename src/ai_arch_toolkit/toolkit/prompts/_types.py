"""Immutable contracts for structured prompt composition."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path

    from ai_arch_toolkit.toolkit.knowledge import KnowledgeRegistry
    from ai_arch_toolkit.toolkit.prompts._layouts import PromptLayout, SectionSpan
    from ai_arch_toolkit.toolkit.resources import (
        Resource,
        ResourcePolicy,
        ResourceResolver,
        ResourceSelector,
        ResourceSerializer,
        SerializerRegistry,
    )

type PromptStability = Literal["static", "session", "request"]

_STABILITIES = frozenset({"static", "session", "request"})


@dataclass(frozen=True, slots=True, init=False)
class PromptSection:
    """A named piece of prompt content with deterministic ordering metadata.

    ``sections`` holds optional subsections; rendering visits a section's own
    content first and then its subsections in canonical preorder.

    ``position`` is accepted as a compatibility alias for ``order`` while
    experimental Nanope consumers migrate to the toolkit API.
    """

    name: str
    content: str
    order: int
    stability: PromptStability
    metadata: Mapping[str, Any] = field(hash=False)
    sections: tuple[PromptSection, ...]

    def __init__(
        self,
        *,
        name: str,
        content: str,
        order: int | None = None,
        stability: PromptStability = "static",
        metadata: Mapping[str, Any] | None = None,
        sections: Sequence[PromptSection] | None = None,
        position: int | None = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("PromptSection.name is required")
        if not isinstance(content, str):
            raise TypeError("PromptSection.content must be a string")
        if order is not None and position is not None:
            raise ValueError("use either order or position, not both")
        if stability not in _STABILITIES:
            choices = ", ".join(sorted(_STABILITIES))
            raise ValueError(f"invalid prompt stability {stability!r}; expected one of: {choices}")

        resolved_order = order if order is not None else position if position is not None else 0
        if not isinstance(resolved_order, int):
            raise TypeError("PromptSection.order must be an integer")

        resolved_sections = tuple(sections) if sections is not None else ()
        for index, child in enumerate(resolved_sections):
            if not isinstance(child, PromptSection):
                raise TypeError(
                    f"PromptSection.sections[{index}] must be a PromptSection, "
                    f"got {type(child).__name__}"
                )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "order", resolved_order)
        object.__setattr__(self, "stability", stability)
        object.__setattr__(self, "sections", resolved_sections)
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("PromptSection.metadata must be a mapping")
        object.__setattr__(
            self, "metadata", _freeze_metadata(metadata if metadata is not None else {})
        )

    @property
    def position(self) -> int:
        """Compatibility alias for the Nanope ``position`` field."""
        return self.order

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        name: str,
        selector: str | ResourceSelector | None = None,
        serialize_as: str | ResourceSerializer | None = None,
        order: int = 0,
        stability: PromptStability = "static",
        metadata: Mapping[str, Any] | None = None,
        sections: Sequence[PromptSection] = (),
        policy: ResourcePolicy | None = None,
        resolver: ResourceResolver | None = None,
    ) -> PromptSection:
        """Load one literal section from a file resource."""
        from ai_arch_toolkit.toolkit.resources import (
            load_resource,
            select_resource,
            serialize_resource_value,
        )

        resource = load_resource(path, policy=policy, resolver=resolver)
        if selector is None and serialize_as is None and resource.text is not None:
            content = resource.text
        else:
            value = select_resource(resource, selector)
            serializer = (
                resolver.serializers.resolve(serialize_as)
                if resolver is not None and serialize_as is not None
                else serialize_as or "text"
            )
            content = serialize_resource_value(value, as_format=serializer)
        resource_metadata = {
            "source": resource.ref.uri,
            "media_type": resource.media_type,
            "resource_fingerprint": resource.fingerprint,
            **dict(metadata or {}),
        }
        return cls(
            name=name,
            content=content,
            order=order,
            stability=stability,
            metadata=resource_metadata,
            sections=sections,
        )

    @classmethod
    def from_resource(
        cls,
        resource: Resource,
        *,
        name: str,
        selector: str | ResourceSelector | None = None,
        serialize_as: str | ResourceSerializer | None = None,
        serializer_registry: SerializerRegistry | None = None,
        order: int = 0,
        stability: PromptStability = "static",
        metadata: Mapping[str, Any] | None = None,
        sections: Sequence[PromptSection] = (),
    ) -> PromptSection:
        """Create one literal section from an already loaded resource."""
        from ai_arch_toolkit.toolkit.resources import (
            Resource,
            select_resource,
            serialize_resource_value,
        )

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
        resource_metadata = {
            "source": resource.ref.uri,
            "media_type": resource.media_type,
            "resource_fingerprint": resource.fingerprint,
            **dict(metadata or {}),
        }
        return cls(
            name=name,
            content=content,
            order=order,
            stability=stability,
            metadata=resource_metadata,
            sections=sections,
        )

    @classmethod
    def from_knowledge(
        cls,
        registry: KnowledgeRegistry,
        keys: Sequence[str],
        *,
        name: str = "knowledge",
        separator: str = "\n\n---\n\n",
        include_names: bool = False,
        order: int = 0,
        stability: PromptStability = "static",
        metadata: Mapping[str, Any] | None = None,
        sections: Sequence[PromptSection] = (),
    ) -> PromptSection:
        """Create a literal section from registered knowledge keys."""
        from ai_arch_toolkit.toolkit.prompts._sources import KnowledgeSource

        resolution = KnowledgeSource(
            registry=registry,
            keys=tuple(keys),
            separator=separator,
            include_names=include_names,
        ).resolve({})
        return cls(
            name=name,
            content=resolution.content,
            order=order,
            stability=stability,
            metadata={
                **dict(metadata or {}),
                "source_provenance": dict(resolution.provenance),
            },
            sections=sections,
        )


def _ordered_sections(sections: Sequence[PromptSection]) -> tuple[PromptSection, ...]:
    """Order one sibling level by ``order``, preserving insertion order on ties."""
    indexed = sorted(enumerate(sections), key=lambda pair: (pair[1].order, pair[0]))
    return tuple(section for _index, section in indexed)


def _walk_sections(
    sections: Sequence[PromptSection],
    *,
    depth: int = 0,
) -> Iterator[tuple[PromptSection, int]]:
    """Yield ``(section, depth)`` in canonical preorder, ordering each sibling level."""
    for section in _ordered_sections(sections):
        yield section, depth
        yield from _walk_sections(section.sections, depth=depth + 1)


@dataclass(frozen=True, slots=True, kw_only=True)
class Prompt:
    """An immutable collection of sections rendered as one prompt."""

    sections: tuple[PromptSection, ...] = ()
    separator: str = "\n\n"

    def __post_init__(self) -> None:
        sections = tuple(self.sections)
        for index, section in enumerate(sections):
            if not isinstance(section, PromptSection):
                raise TypeError(
                    f"Prompt.sections[{index}] must be a PromptSection, "
                    f"got {type(section).__name__}"
                )
        object.__setattr__(self, "sections", sections)
        if not isinstance(self.separator, str):
            raise TypeError("Prompt.separator must be a string")

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        name: str = "prompt",
        stability: PromptStability = "static",
    ) -> Prompt:
        """Create a one-section literal prompt."""
        return cls(sections=(PromptSection(name=name, content=text, stability=stability),))

    @classmethod
    def from_sections(
        cls,
        *sections: PromptSection,
        separator: str = "\n\n",
    ) -> Prompt:
        """Create a prompt from positional sections."""
        return cls(sections=sections, separator=separator)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        name: str = "prompt",
        selector: str | ResourceSelector | None = None,
        serialize_as: str | ResourceSerializer | None = None,
        stability: PromptStability = "static",
        policy: ResourcePolicy | None = None,
        resolver: ResourceResolver | None = None,
    ) -> Prompt:
        """Create a one-section literal prompt from a file resource."""
        section = PromptSection.from_file(
            path,
            name=name,
            selector=selector,
            serialize_as=serialize_as,
            stability=stability,
            policy=policy,
            resolver=resolver,
        )
        return cls(sections=(section,))

    @classmethod
    def from_resource(
        cls,
        resource: Resource,
        *,
        name: str = "prompt",
        selector: str | ResourceSelector | None = None,
        serialize_as: str | ResourceSerializer | None = None,
        serializer_registry: SerializerRegistry | None = None,
        stability: PromptStability = "static",
    ) -> Prompt:
        """Create a one-section literal prompt from an in-memory resource."""
        section = PromptSection.from_resource(
            resource,
            name=name,
            selector=selector,
            serialize_as=serialize_as,
            serializer_registry=serializer_registry,
            stability=stability,
        )
        return cls(sections=(section,))

    def render(
        self,
        *,
        layout: str | PromptLayout | None = None,
        separator: str | None = None,
    ) -> RenderedPrompt:
        """Render this prompt using a built-in name or layout object."""
        from ai_arch_toolkit.toolkit.prompts._layouts import layout_from_name
        from ai_arch_toolkit.toolkit.prompts._render import render_prompt

        if separator is None:
            return render_prompt(self, layout=layout)
        if layout is not None and not isinstance(layout, str):
            raise ValueError("separator cannot be combined with a layout object")
        active_layout = layout_from_name(layout or "text", separator=separator)
        return render_prompt(self, layout=active_layout)


@dataclass(frozen=True, slots=True, kw_only=True)
class RenderedPrompt:
    """Exact rendered prompt plus provenance and cache-layout diagnostics."""

    text: str
    sections: tuple[PromptSection, ...]
    fingerprint: str
    stable_prefix_end: int | None = None
    section_spans: tuple[SectionSpan, ...] = ()
    layout: str = "text"
    provenance: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", _freeze_metadata(self.provenance))

    @property
    def section_names(self) -> tuple[str, ...]:
        """Return section names in canonical preorder render order."""
        return tuple(section.name for section, _depth in _walk_sections(self.sections))

    @property
    def stable_prefix(self) -> str:
        """Return the initial static prefix suitable for provider cache planning."""
        if self.stable_prefix_end is None:
            return ""
        return self.text[: self.stable_prefix_end]

    @property
    def system(self) -> str:
        """Compatibility alias for Nanope's former rendered prompt contract."""
        return self.text

    def section_text(self, name: str) -> str:
        """Return the exact rendered slice occupied by a named section."""
        for span in self.section_spans:
            if span.name == name:
                return self.text[span.start : span.end]
        raise KeyError(f"rendered prompt has no section named {name!r}")


def prompt_from_sections(
    sections: Sequence[PromptSection],
    *,
    separator: str = "\n\n",
) -> Prompt:
    """Build a prompt from any finite sequence of sections."""
    return Prompt(sections=tuple(sections), separator=separator)


def _freeze_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recursively freeze built-in metadata containers."""
    return _freeze_value(metadata, seen=set())


def _freeze_value(value: Any, *, seen: set[int]) -> Any:
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise ValueError("PromptSection.metadata cannot contain cycles")
        seen.add(identity)
        try:
            frozen: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("PromptSection.metadata keys must be strings")
                frozen[key] = _freeze_value(item, seen=seen)
            return MappingProxyType(frozen)
        finally:
            seen.remove(identity)
    if isinstance(value, list | tuple):
        identity = id(value)
        if identity in seen:
            raise ValueError("PromptSection.metadata cannot contain cycles")
        seen.add(identity)
        try:
            return tuple(_freeze_value(item, seen=seen) for item in value)
        finally:
            seen.remove(identity)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_value(item, seen=seen) for item in value)
    return value


__all__ = [
    "Prompt",
    "PromptSection",
    "PromptStability",
    "RenderedPrompt",
    "prompt_from_sections",
]
