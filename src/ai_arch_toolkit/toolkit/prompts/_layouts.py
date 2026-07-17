"""Prompt layouts for deterministic section serialization."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from ai_arch_toolkit.toolkit.prompts._types import PromptSection, _ordered_sections


@dataclass(frozen=True, slots=True, kw_only=True)
class SectionSpan:
    """Offsets occupied by one section in rendered prompt text.

    A parent section's span covers its whole subtree; ``content_start`` and
    ``content_end`` bound only the section's own content. ``depth`` is the
    section's nesting level in canonical preorder.
    """

    name: str
    start: int
    end: int
    content_start: int | None = None
    content_end: int | None = None
    depth: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("SectionSpan.name is required")
        offsets = (self.start, self.end, self.content_start, self.content_end)
        if any(
            value is not None and (not isinstance(value, int) or isinstance(value, bool))
            for value in offsets
        ):
            raise TypeError("section span offsets must be integers or None")
        if not isinstance(self.depth, int) or isinstance(self.depth, bool):
            raise TypeError("SectionSpan.depth must be an integer")
        if self.depth < 0:
            raise ValueError("SectionSpan.depth must be non-negative")
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
    """Serialize already ordered prompt sections.

    Layouts receive top-level sections in render order and are responsible for
    ordering and rendering each section's subsections (canonical preorder),
    returning one span per tree node.
    """

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
    """Join literal section content with configurable separators.

    Subsections are flattened in preorder; text has no hierarchy affordance.
    """

    separator: str | SeparatorPolicy = "\n\n"
    name: str = field(default="text", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.separator, str | SeparatorPolicy):
            raise TypeError("TextLayout.separator must be a string or SeparatorPolicy")

    def render(self, sections: Sequence[PromptSection]) -> LayoutResult:
        parts: list[str] = []
        cursor = 0
        previous: PromptSection | None = None

        def emit(section: PromptSection, depth: int) -> list[SectionSpan]:
            nonlocal cursor, previous
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
            content_end = cursor
            after = _after(self.separator, section)
            parts.append(after)
            cursor += len(after)
            previous = section
            child_spans: list[SectionSpan] = []
            for child in _ordered_sections(section.sections):
                child_spans.extend(emit(child, depth + 1))
            end = child_spans[-1].end if child_spans else content_end
            span = SectionSpan(
                name=section.name,
                start=start,
                end=end,
                content_start=start,
                content_end=content_end,
                depth=depth,
            )
            return [span, *child_spans]

        spans: list[SectionSpan] = []
        for section in sections:
            spans.extend(emit(section, 0))
        return LayoutResult(text="".join(parts), spans=tuple(spans), layout=self.name)


@dataclass(frozen=True, slots=True, kw_only=True)
class MarkdownLayout:
    """Render sections as titled Markdown blocks.

    Subsections deepen the heading by one level each; rendering fails when a
    subsection would exceed heading level 6.
    """

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
        parts: list[str] = []
        cursor = 0
        previous: PromptSection | None = None

        def emit(section: PromptSection, depth: int) -> list[SectionSpan]:
            nonlocal cursor, previous
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
            title = section.metadata.get("title", section.name)
            if not isinstance(title, str):
                raise TypeError(f"section {section.name!r} metadata title must be a string")
            prefix = ""
            if self.include_headings:
                level = self.heading_level + depth
                if level > 6:
                    raise ValueError(
                        f"markdown heading level {level} for section {section.name!r} "
                        "exceeds 6; reduce heading_level or nesting depth"
                    )
                prefix = f"{'#' * level} {title}\n\n"
            block = prefix + section.content
            start = cursor
            content_start = start + len(prefix)
            parts.append(block)
            cursor += len(block)
            content_end = cursor
            after = _after(self.separator, section)
            parts.append(after)
            cursor += len(after)
            previous = section
            child_spans: list[SectionSpan] = []
            for child in _ordered_sections(section.sections):
                child_spans.extend(emit(child, depth + 1))
            end = child_spans[-1].end if child_spans else content_end
            span = SectionSpan(
                name=section.name,
                start=start,
                end=end,
                content_start=content_start,
                content_end=content_end,
                depth=depth,
            )
            return [span, *child_spans]

        spans: list[SectionSpan] = []
        for section in sections:
            spans.extend(emit(section, 0))
        return LayoutResult(text="".join(parts), spans=tuple(spans), layout=self.name)


_XML_TAG = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True, kw_only=True)
class XmlLayout:
    """Render sections as escaped XML elements; subsections nest as child elements."""

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

    def _serialize_element(self, section: PromptSection) -> str:
        element = ET.Element(self.section_tag, {"name": section.name})
        if self.include_stability:
            element.set("stability", section.stability)
        for name in self.metadata_attributes:
            if name in section.metadata:
                value = section.metadata[name]
                if not isinstance(value, str | int | float | bool):
                    raise TypeError(
                        f"section {section.name!r} metadata attribute {name!r} must be a scalar"
                    )
                element.set(name, str(value).lower() if isinstance(value, bool) else str(value))
        element.text = section.content
        return ET.tostring(element, encoding="unicode", short_empty_elements=False)

    def render(self, sections: Sequence[PromptSection]) -> LayoutResult:
        root = ET.Element(self.root_tag)
        empty_root = ET.tostring(root, encoding="unicode", short_empty_elements=False)
        close_root = f"</{self.root_tag}>"
        open_root = empty_root[: -len(close_root)]
        close_section = f"</{self.section_tag}>"
        parts = [open_root]
        cursor = len(open_root)

        def emit(section: PromptSection, depth: int) -> list[SectionSpan]:
            nonlocal cursor
            serialized = self._serialize_element(section)
            open_and_text = serialized[: -len(close_section)]
            start = cursor
            content_start = start + serialized.index(">") + 1
            parts.append(open_and_text)
            cursor += len(open_and_text)
            content_end = cursor
            child_spans: list[SectionSpan] = []
            children = _ordered_sections(section.sections)
            for child in children:
                parts.append(self.separator)
                cursor += len(self.separator)
                child_spans.extend(emit(child, depth + 1))
            if children:
                parts.append(self.separator)
                cursor += len(self.separator)
            parts.append(close_section)
            cursor += len(close_section)
            span = SectionSpan(
                name=section.name,
                start=start,
                end=cursor,
                content_start=content_start,
                content_end=content_end,
                depth=depth,
            )
            return [span, *child_spans]

        spans: list[SectionSpan] = []
        if sections:
            parts.append(self.separator)
            cursor += len(self.separator)
        for index, section in enumerate(sections):
            spans.extend(emit(section, 0))
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
    """Render sections as an ordered JSON array; subsections nest recursively."""

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

    def _dumps(self, value: str) -> str:
        return json.dumps(value, ensure_ascii=self.ensure_ascii)

    def render(self, sections: Sequence[PromptSection]) -> LayoutResult:
        if self.mode == "object":
            return self._render_object(sections)
        compact = self.indent is None
        key_joiner = ":" if compact else ": "
        item_joiner = "," if compact else ", "
        if self.indent is None:
            prefix, separator, suffix, item_prefix = "[", ",", "]", ""
        else:
            prefix, separator, suffix, item_prefix = "[\n", ",\n", "\n]", " " * self.indent
        if not sections and self.indent is not None:
            return LayoutResult(text="[]", spans=(), layout=self.name)
        parts = [prefix]
        cursor = len(prefix)

        def emit(section: PromptSection, depth: int, lead: str) -> list[SectionSpan]:
            nonlocal cursor
            start = cursor
            fields = [
                f"{self._dumps('name')}{key_joiner}{self._dumps(section.name)}",
                f"{self._dumps('content')}{key_joiner}{self._dumps(section.content)}",
            ]
            if self.include_stability:
                fields.append(
                    f"{self._dumps('stability')}{key_joiner}{self._dumps(section.stability)}"
                )
            opening = lead + "{" + item_joiner.join(fields)
            parts.append(opening)
            cursor += len(opening)
            child_spans: list[SectionSpan] = []
            children = _ordered_sections(section.sections)
            if children:
                sections_key = f"{item_joiner}{self._dumps('sections')}{key_joiner}["
                parts.append(sections_key)
                cursor += len(sections_key)
                for child_index, child in enumerate(children):
                    if child_index:
                        parts.append(item_joiner)
                        cursor += len(item_joiner)
                    child_spans.extend(emit(child, depth + 1, ""))
                parts.append("]")
                cursor += 1
            parts.append("}")
            cursor += 1
            span = SectionSpan(name=section.name, start=start, end=cursor, depth=depth)
            return [span, *child_spans]

        spans: list[SectionSpan] = []
        for index, section in enumerate(sections):
            if index:
                parts.append(separator)
                cursor += len(separator)
            spans.extend(emit(section, 0, item_prefix))
        parts.append(suffix)
        return LayoutResult(text="".join(parts), spans=tuple(spans), layout=self.name)

    def _render_object(self, sections: Sequence[PromptSection]) -> LayoutResult:
        compact = self.indent is None
        key_joiner = ":" if compact else ": "
        item_joiner = "," if compact else ", "
        parts: list[str] = []
        cursor = 0

        def write(text: str) -> None:
            nonlocal cursor
            parts.append(text)
            cursor += len(text)

        def emit_entry(section: PromptSection, depth: int) -> list[SectionSpan]:
            start = cursor
            write(f"{self._dumps(section.name)}{key_joiner}")
            child_spans = emit_value(section, depth)
            span = SectionSpan(name=section.name, start=start, end=cursor, depth=depth)
            return [span, *child_spans]

        def emit_value(section: PromptSection, depth: int) -> list[SectionSpan]:
            children = _ordered_sections(section.sections)
            if not children and not self.include_stability:
                write(self._dumps(section.content))
                return []
            write("{" + f"{self._dumps('content')}{key_joiner}{self._dumps(section.content)}")
            if self.include_stability:
                write(f"{item_joiner}{self._dumps('stability')}{key_joiner}")
                write(self._dumps(section.stability))
            child_spans: list[SectionSpan] = []
            if children:
                write(f"{item_joiner}{self._dumps('sections')}{key_joiner}" + "{")
                for index, child in enumerate(children):
                    if index:
                        write(item_joiner)
                    child_spans.extend(emit_entry(child, depth + 1))
                write("}")
            write("}")
            return child_spans

        spans: list[SectionSpan] = []
        if compact:
            write("{")
            for index, section in enumerate(sections):
                if index:
                    write(",")
                spans.extend(emit_entry(section, 0))
            write("}")
        else:
            if not sections:
                return LayoutResult(text="{}", spans=(), layout=self.name)
            padding = " " * self.indent  # type: ignore[operator]
            write("{\n")
            for index, section in enumerate(sections):
                if index:
                    write(",\n")
                write(padding)
                spans.extend(emit_entry(section, 0))
            write("\n}")
        return LayoutResult(text="".join(parts), spans=tuple(spans), layout=self.name)


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
