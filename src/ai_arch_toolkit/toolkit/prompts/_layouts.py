"""Prompt layouts for deterministic section serialization."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from ai_arch_toolkit.toolkit.prompts._types import PromptSection


@dataclass(frozen=True, slots=True, kw_only=True)
class SectionSpan:
    """Offsets occupied by one section in rendered prompt text."""

    name: str
    start: int
    end: int
    content_start: int | None = None
    content_end: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("SectionSpan.name is required")
        offsets = (self.start, self.end, self.content_start, self.content_end)
        if any(
            value is not None and (not isinstance(value, int) or isinstance(value, bool))
            for value in offsets
        ):
            raise TypeError("section span offsets must be integers or None")
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid section span offsets")
        if (self.content_start is None) != (self.content_end is None):
            raise ValueError("content_start and content_end must both be set or both be None")
        if self.content_start is not None and not (
            self.start <= self.content_start <= self.content_end <= self.end  # type: ignore[operator]
        ):
            raise ValueError("content offsets must be contained by the section span")


@dataclass(frozen=True, slots=True, kw_only=True)
class LayoutResult:
    """Text and section offsets produced by a prompt layout."""

    text: str
    spans: tuple[SectionSpan, ...]
    layout: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("LayoutResult.text must be a string")
        spans = tuple(self.spans)
        if not all(isinstance(span, SectionSpan) for span in spans):
            raise TypeError("LayoutResult.spans must contain SectionSpan values")
        if not isinstance(self.layout, str) or not self.layout:
            raise ValueError("LayoutResult.layout is required")
        object.__setattr__(self, "spans", spans)


class PromptLayout(Protocol):
    """Serialize already ordered prompt sections."""

    @property
    def name(self) -> str:
        """Stable public layout name."""
        ...

    def render(self, sections: Sequence[PromptSection]) -> LayoutResult:
        """Render ordered sections."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class SeparatorPolicy:
    """Choose separators globally or for named section boundaries."""

    default: str = "\n\n"
    between: Mapping[tuple[str, str], str] = field(default_factory=dict)
    before: Mapping[str, str] = field(default_factory=dict)
    after: Mapping[str, str] = field(default_factory=dict)
    resolver: Callable[[PromptSection, PromptSection], str] | None = field(
        default=None, compare=False, hash=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.default, str):
            raise TypeError("SeparatorPolicy.default must be a string")
        if not isinstance(self.between, Mapping):
            raise TypeError("SeparatorPolicy.between must be a mapping")
        if not isinstance(self.before, Mapping) or not isinstance(self.after, Mapping):
            raise TypeError("SeparatorPolicy.before and after must be mappings")
        if self.resolver is not None and not callable(self.resolver):
            raise TypeError("SeparatorPolicy.resolver must be callable or None")
        normalized: dict[tuple[str, str], str] = {}
        for boundary, separator in self.between.items():
            if (
                not isinstance(boundary, tuple)
                or len(boundary) != 2
                or not all(isinstance(name, str) for name in boundary)
            ):
                raise TypeError("separator boundaries must be (previous_name, next_name) tuples")
            if not isinstance(separator, str):
                raise TypeError("boundary separators must be strings")
            normalized[boundary] = separator
        object.__setattr__(self, "between", MappingProxyType(normalized))
        object.__setattr__(self, "before", MappingProxyType(_named_separators(self.before)))
        object.__setattr__(self, "after", MappingProxyType(_named_separators(self.after)))

    def separator(self, previous: PromptSection, current: PromptSection) -> str:
        """Return the separator for one adjacent pair."""
        if self.resolver is not None:
            separator = self.resolver(previous, current)
            if not isinstance(separator, str):
                raise TypeError("SeparatorPolicy.resolver must return a string")
            return separator
        return self.between.get((previous.name, current.name), self.default)


def _named_separators(values: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for name, separator in values.items():
        if not isinstance(name, str) or not isinstance(separator, str):
            raise TypeError("named section separators must map strings to strings")
        normalized[name] = separator
    return normalized


def _before(separator: str | SeparatorPolicy, section: PromptSection) -> str:
    return "" if isinstance(separator, str) else separator.before.get(section.name, "")


def _after(separator: str | SeparatorPolicy, section: PromptSection) -> str:
    return "" if isinstance(separator, str) else separator.after.get(section.name, "")


@dataclass(frozen=True, slots=True, kw_only=True)
class TextLayout:
    """Join literal section content with configurable separators."""

    separator: str | SeparatorPolicy = "\n\n"
    name: str = field(default="text", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.separator, str | SeparatorPolicy):
            raise TypeError("TextLayout.separator must be a string or SeparatorPolicy")

    def render(self, sections: Sequence[PromptSection]) -> LayoutResult:
        parts: list[str] = []
        spans: list[SectionSpan] = []
        cursor = 0
        previous: PromptSection | None = None
        for section in sections:
            if previous is not None:
                separator = (
                    self.separator
                    if isinstance(self.separator, str)
                    else self.separator.separator(previous, section)
                )
                parts.append(separator)
                cursor += len(separator)
            before = _before(self.separator, section)
            parts.append(before)
            cursor += len(before)
            start = cursor
            parts.append(section.content)
            cursor += len(section.content)
            spans.append(
                SectionSpan(
                    name=section.name,
                    start=start,
                    end=cursor,
                    content_start=start,
                    content_end=cursor,
                )
            )
            after = _after(self.separator, section)
            parts.append(after)
            cursor += len(after)
            previous = section
        return LayoutResult(text="".join(parts), spans=tuple(spans), layout=self.name)


@dataclass(frozen=True, slots=True, kw_only=True)
class MarkdownLayout:
    """Render sections as titled Markdown blocks."""

    heading_level: int = 2
    separator: str | SeparatorPolicy = "\n\n"
    include_headings: bool = True
    name: str = field(default="markdown", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.heading_level, int) or isinstance(self.heading_level, bool):
            raise TypeError("MarkdownLayout.heading_level must be an integer")
        if not 1 <= self.heading_level <= 6:
            raise ValueError("MarkdownLayout.heading_level must be between 1 and 6")
        if not isinstance(self.separator, str | SeparatorPolicy):
            raise TypeError("MarkdownLayout.separator must be a string or SeparatorPolicy")
        if not isinstance(self.include_headings, bool):
            raise TypeError("MarkdownLayout.include_headings must be a boolean")

    def render(self, sections: Sequence[PromptSection]) -> LayoutResult:
        blocks: list[str] = []
        spans: list[SectionSpan] = []
        cursor = 0
        previous: PromptSection | None = None
        for section in sections:
            if previous is not None:
                separator = (
                    self.separator
                    if isinstance(self.separator, str)
                    else self.separator.separator(previous, section)
                )
                blocks.append(separator)
                cursor += len(separator)
            before = _before(self.separator, section)
            blocks.append(before)
            cursor += len(before)
            title = section.metadata.get("title", section.name)
            if not isinstance(title, str):
                raise TypeError(f"section {section.name!r} metadata title must be a string")
            prefix = f"{'#' * self.heading_level} {title}\n\n" if self.include_headings else ""
            block = prefix + section.content
            start = cursor
            content_start = start + len(prefix)
            blocks.append(block)
            cursor += len(block)
            spans.append(
                SectionSpan(
                    name=section.name,
                    start=start,
                    end=cursor,
                    content_start=content_start,
                    content_end=cursor,
                )
            )
            after = _after(self.separator, section)
            blocks.append(after)
            cursor += len(after)
            previous = section
        return LayoutResult(text="".join(blocks), spans=tuple(spans), layout=self.name)


_XML_TAG = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True, kw_only=True)
class XmlLayout:
    """Render sections as escaped XML elements."""

    root_tag: str = "prompt"
    section_tag: str = "section"
    separator: str = "\n"
    include_stability: bool = False
    metadata_attributes: tuple[str, ...] = ()
    name: str = field(default="xml", init=False)

    def __post_init__(self) -> None:
        for field_name, value in (("root_tag", self.root_tag), ("section_tag", self.section_tag)):
            if not isinstance(value, str) or not _XML_TAG.fullmatch(value):
                raise ValueError(f"XmlLayout.{field_name} is not a valid XML tag: {value!r}")
        if not isinstance(self.separator, str):
            raise TypeError("XmlLayout.separator must be a string")
        if not isinstance(self.include_stability, bool):
            raise TypeError("XmlLayout.include_stability must be a boolean")
        attributes = tuple(self.metadata_attributes)
        if not all(isinstance(name, str) and name for name in attributes):
            raise ValueError("XmlLayout.metadata_attributes must contain non-empty strings")
        object.__setattr__(self, "metadata_attributes", attributes)

    def render(self, sections: Sequence[PromptSection]) -> LayoutResult:
        root = ET.Element(self.root_tag)
        empty_root = ET.tostring(root, encoding="unicode", short_empty_elements=False)
        close_root = f"</{self.root_tag}>"
        open_root = empty_root[: -len(close_root)]
        parts = [open_root]
        spans: list[SectionSpan] = []
        cursor = len(open_root)
        if sections:
            parts.append(self.separator)
            cursor += len(self.separator)
        for index, section in enumerate(sections):
            element = ET.Element(self.section_tag, {"name": section.name})
            if self.include_stability:
                element.set("stability", section.stability)
            for name in self.metadata_attributes:
                if name in section.metadata:
                    value = section.metadata[name]
                    if not isinstance(value, str | int | float | bool):
                        raise TypeError(
                            f"section {section.name!r} metadata attribute {name!r} "
                            "must be a scalar"
                        )
                    element.set(
                        name, str(value).lower() if isinstance(value, bool) else str(value)
                    )
            element.text = section.content
            serialized = ET.tostring(element, encoding="unicode", short_empty_elements=False)
            start = cursor
            content_start = start + serialized.index(">") + 1
            content_end = start + serialized.rindex(f"</{self.section_tag}>")
            parts.append(serialized)
            cursor += len(serialized)
            spans.append(
                SectionSpan(
                    name=section.name,
                    start=start,
                    end=cursor,
                    content_start=content_start,
                    content_end=content_end,
                )
            )
            if index < len(sections) - 1:
                parts.append(self.separator)
                cursor += len(self.separator)
        if sections:
            parts.append(self.separator)
            cursor += len(self.separator)
        parts.append(close_root)
        return LayoutResult(text="".join(parts), spans=tuple(spans), layout=self.name)


@dataclass(frozen=True, slots=True, kw_only=True)
class JsonLayout:
    """Render sections as an ordered JSON array."""

    indent: int | None = 2
    include_stability: bool = False
    ensure_ascii: bool = False
    mode: str = "array"
    name: str = field(default="json", init=False)

    def __post_init__(self) -> None:
        if self.indent is not None and (
            not isinstance(self.indent, int) or isinstance(self.indent, bool) or self.indent < 0
        ):
            raise ValueError("JsonLayout.indent must be a non-negative integer or None")
        if not isinstance(self.include_stability, bool):
            raise TypeError("JsonLayout.include_stability must be a boolean")
        if not isinstance(self.ensure_ascii, bool):
            raise TypeError("JsonLayout.ensure_ascii must be a boolean")
        if self.mode not in {"array", "object"}:
            raise ValueError("JsonLayout.mode must be 'array' or 'object'")

    def render(self, sections: Sequence[PromptSection]) -> LayoutResult:
        if self.mode == "object":
            return self._render_object(sections)
        if self.indent is None:
            prefix, separator, suffix, item_prefix = "[", ",", "]", ""
        else:
            prefix, separator, suffix, item_prefix = "[\n", ",\n", "\n]", " " * self.indent
        parts = [prefix]
        spans: list[SectionSpan] = []
        cursor = len(prefix)
        for index, section in enumerate(sections):
            if index:
                parts.append(separator)
                cursor += len(separator)
            payload: dict[str, str] = {"name": section.name, "content": section.content}
            if self.include_stability:
                payload["stability"] = section.stability
            serialized = json.dumps(
                payload,
                ensure_ascii=self.ensure_ascii,
                separators=(",", ":") if self.indent is None else None,
            )
            if item_prefix:
                serialized = item_prefix + serialized
            start = cursor
            parts.append(serialized)
            cursor += len(serialized)
            spans.append(SectionSpan(name=section.name, start=start, end=cursor))
        if not sections and self.indent is not None:
            return LayoutResult(text="[]", spans=(), layout=self.name)
        parts.append(suffix)
        return LayoutResult(text="".join(parts), spans=tuple(spans), layout=self.name)

    def _render_object(self, sections: Sequence[PromptSection]) -> LayoutResult:
        compact = self.indent is None
        value_separators = (",", ":") if compact else None
        joiner = ":" if compact else ": "
        entries: list[str] = []
        for section in sections:
            value: str | dict[str, str] = section.content
            if self.include_stability:
                value = {"content": section.content, "stability": section.stability}
            key = json.dumps(section.name, ensure_ascii=self.ensure_ascii)
            serialized_value = json.dumps(
                value, ensure_ascii=self.ensure_ascii, separators=value_separators
            )
            entries.append(f"{key}{joiner}{serialized_value}")
        if compact:
            text = "{" + ",".join(entries) + "}"
        elif entries:
            padding = " " * self.indent  # type: ignore[operator]
            text = "{\n" + ",\n".join(padding + entry for entry in entries) + "\n}"
        else:
            text = "{}"
        spans: list[SectionSpan] = []
        cursor = 0
        for section, entry in zip(sections, entries, strict=True):
            start = text.find(entry, cursor)
            spans.append(SectionSpan(name=section.name, start=start, end=start + len(entry)))
            cursor = start + len(entry)
        return LayoutResult(text=text, spans=tuple(spans), layout=self.name)


def layout_from_name(name: str, *, separator: str = "\n\n") -> PromptLayout:
    """Create a built-in layout from its short name."""
    if not isinstance(name, str):
        raise TypeError("prompt layout name must be a string")
    normalized = name.lower()
    if normalized == "text":
        return TextLayout(separator=separator)
    if normalized == "markdown":
        return MarkdownLayout(separator=separator)
    if normalized == "xml":
        return XmlLayout()
    if normalized == "json":
        return JsonLayout()
    raise ValueError("unknown prompt layout; expected one of: json, markdown, text, xml")


__all__ = [
    "JsonLayout",
    "LayoutResult",
    "MarkdownLayout",
    "PromptLayout",
    "SectionSpan",
    "SeparatorPolicy",
    "TextLayout",
    "XmlLayout",
    "layout_from_name",
]
