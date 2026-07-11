"""Immutable contracts for structured prompt composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

type PromptStability = Literal["static", "session", "request"]

_STABILITIES = frozenset({"static", "session", "request"})


@dataclass(frozen=True, slots=True, init=False)
class PromptSection:
    """A named piece of prompt content with deterministic ordering metadata.

    ``position`` is accepted as a compatibility alias for ``order`` while
    experimental Nanope consumers migrate to the toolkit API.
    """

    name: str
    content: str
    order: int
    stability: PromptStability
    metadata: Mapping[str, Any] = field(hash=False)

    def __init__(
        self,
        *,
        name: str,
        content: str,
        order: int | None = None,
        stability: PromptStability = "static",
        metadata: Mapping[str, Any] | None = None,
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

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "order", resolved_order)
        object.__setattr__(self, "stability", stability)
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("PromptSection.metadata must be a mapping")
        object.__setattr__(
            self, "metadata", _freeze_metadata(metadata if metadata is not None else {})
        )

    @property
    def position(self) -> int:
        """Compatibility alias for the Nanope ``position`` field."""
        return self.order


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


@dataclass(frozen=True, slots=True, kw_only=True)
class RenderedPrompt:
    """Exact rendered prompt plus provenance and cache-layout diagnostics."""

    text: str
    sections: tuple[PromptSection, ...]
    fingerprint: str
    stable_prefix_end: int | None = None

    @property
    def section_names(self) -> tuple[str, ...]:
        """Return section names in render order."""
        return tuple(section.name for section in self.sections)

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
