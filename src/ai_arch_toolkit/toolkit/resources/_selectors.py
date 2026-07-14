"""Selectors for extracting values and text fragments from resources."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from ai_arch_toolkit.toolkit.resources._errors import ResourceSelectorError
from ai_arch_toolkit.toolkit.resources._types import Resource

_STRUCTURED_MEDIA_TYPES = frozenset(
    {
        "application/json",
        "application/toml",
        "application/yaml",
        "application/x-yaml",
        "text/yaml",
    }
)


class ResourceSelector(Protocol):
    """Select a value or fragment from a loaded resource."""

    def select(self, resource: Resource) -> Any:
        """Return the selected value."""
        ...


@dataclass(frozen=True, slots=True)
class IdentitySelector:
    """Return the parsed resource value unchanged."""

    def select(self, resource: Resource) -> Any:
        return resource.data


@dataclass(frozen=True, slots=True)
class JsonPointer:
    """Select a structured value using RFC 6901 JSON Pointer syntax."""

    pointer: str

    def __post_init__(self) -> None:
        if not isinstance(self.pointer, str):
            raise TypeError("JsonPointer.pointer must be a string")

    def select(self, resource: Resource) -> Any:
        if self.pointer == "":
            return resource.data
        if not self.pointer.startswith("/"):
            raise ResourceSelectorError(
                f"invalid JSON Pointer {self.pointer!r}; expected an empty string or leading '/'"
            )
        current = resource.data
        for raw_token in self.pointer[1:].split("/"):
            token = _decode_pointer_token(raw_token, pointer=self.pointer)
            if isinstance(current, dict):
                if token not in current:
                    available = ", ".join(map(str, current.keys())) or "(none)"
                    raise ResourceSelectorError(
                        f"JSON Pointer {self.pointer!r} did not find key {token!r} in "
                        f"{resource.ref.uri!r}; available keys: {available}"
                    )
                current = current[token]
            elif isinstance(current, list):
                if (
                    token == "-"
                    or not token.isdigit()
                    or (len(token) > 1 and token.startswith("0"))
                ):
                    raise ResourceSelectorError(
                        f"JSON Pointer {self.pointer!r} has invalid array index {token!r}"
                    )
                index = int(token)
                if index >= len(current):
                    raise ResourceSelectorError(
                        f"JSON Pointer {self.pointer!r} index {index} is outside array of "
                        f"length {len(current)}"
                    )
                current = current[index]
            else:
                raise ResourceSelectorError(
                    f"JSON Pointer {self.pointer!r} cannot traverse {type(current).__name__} "
                    f"at token {token!r}"
                )
        return current


def _decode_pointer_token(token: str, *, pointer: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(token):
        if token[index] != "~":
            result.append(token[index])
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
            raise ResourceSelectorError(f"invalid escape in JSON Pointer {pointer!r}")
        result.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(result)


@dataclass(frozen=True, slots=True, kw_only=True)
class MarkdownHeading:
    """Select one Markdown ATX heading section by exact title."""

    heading: str
    occurrence: int | None = None
    include_heading: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.heading, str) or not self.heading:
            raise ValueError("MarkdownHeading.heading is required")
        if self.occurrence is not None and not isinstance(self.occurrence, int):
            raise TypeError("MarkdownHeading.occurrence must be an integer or None")
        if self.occurrence is not None and self.occurrence < 1:
            raise ValueError("MarkdownHeading.occurrence must be at least 1")
        if not isinstance(self.include_heading, bool):
            raise TypeError("MarkdownHeading.include_heading must be a boolean")

    def select(self, resource: Resource) -> str:
        if resource.text is None:
            raise ResourceSelectorError(
                f"Markdown heading selection requires text: {resource.ref.uri!r}"
            )
        lines = resource.text.splitlines(keepends=True)
        matches: list[tuple[int, int]] = []
        heading_pattern = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*(?:\r?\n)?$")
        for index, line in enumerate(lines):
            match = heading_pattern.match(line)
            if match and match.group(2).strip() == self.heading:
                matches.append((index, len(match.group(1))))
        if not matches:
            raise ResourceSelectorError(
                f"Markdown heading {self.heading!r} was not found in {resource.ref.uri!r}"
            )
        if self.occurrence is None and len(matches) > 1:
            raise ResourceSelectorError(
                f"Markdown heading {self.heading!r} is ambiguous in {resource.ref.uri!r}; "
                f"found {len(matches)} occurrences"
            )
        selected_index = (self.occurrence or 1) - 1
        if selected_index >= len(matches):
            raise ResourceSelectorError(
                f"Markdown heading {self.heading!r} occurrence {selected_index + 1} "
                f"does not exist; found {len(matches)}"
            )
        start, level = matches[selected_index]
        end = len(lines)
        for index in range(start + 1, len(lines)):
            match = heading_pattern.match(lines[index])
            if match and len(match.group(1)) <= level:
                end = index
                break
        content_start = start if self.include_heading else start + 1
        return "".join(lines[content_start:end]).strip("\r\n")


@dataclass(frozen=True, slots=True, kw_only=True)
class LineRange:
    """Select a one-based inclusive line range from a text resource."""

    start: int
    end: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.start, int) or isinstance(self.start, bool):
            raise TypeError("LineRange.start must be an integer")
        if self.end is not None and (not isinstance(self.end, int) or isinstance(self.end, bool)):
            raise TypeError("LineRange.end must be an integer or None")
        if self.start < 1:
            raise ValueError("LineRange.start must be at least 1")
        if self.end is not None and self.end < self.start:
            raise ValueError("LineRange.end must be greater than or equal to start")

    def select(self, resource: Resource) -> str:
        if resource.text is None:
            raise ResourceSelectorError(f"line selection requires text: {resource.ref.uri!r}")
        lines = resource.text.splitlines(keepends=True)
        if self.start > len(lines):
            raise ResourceSelectorError(
                f"line {self.start} is outside {resource.ref.uri!r}, which has {len(lines)} lines"
            )
        end = self.end or len(lines)
        return "".join(lines[self.start - 1 : end])


@dataclass(frozen=True, slots=True, kw_only=True)
class NamedBlock:
    """Select text between exact start and end marker lines."""

    start_marker: str
    end_marker: str
    include_markers: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.start_marker, str) or not self.start_marker:
            raise ValueError("NamedBlock.start_marker is required")
        if not isinstance(self.end_marker, str) or not self.end_marker:
            raise ValueError("NamedBlock.end_marker is required")
        if self.start_marker == self.end_marker:
            raise ValueError("NamedBlock markers must be different")
        if not isinstance(self.include_markers, bool):
            raise TypeError("NamedBlock.include_markers must be a boolean")

    def select(self, resource: Resource) -> str:
        if resource.text is None:
            raise ResourceSelectorError(
                f"named-block selection requires text: {resource.ref.uri!r}"
            )
        lines = resource.text.splitlines(keepends=True)
        starts = [
            index for index, line in enumerate(lines) if line.rstrip("\r\n") == self.start_marker
        ]
        if len(starts) != 1:
            raise ResourceSelectorError(
                f"expected exactly one start marker {self.start_marker!r} in "
                f"{resource.ref.uri!r}; found {len(starts)}"
            )
        start = starts[0]
        ends = [
            index
            for index, line in enumerate(lines[start + 1 :], start=start + 1)
            if line.rstrip("\r\n") == self.end_marker
        ]
        if len(ends) != 1:
            raise ResourceSelectorError(
                f"expected exactly one end marker {self.end_marker!r} after start marker in "
                f"{resource.ref.uri!r}; found {len(ends)}"
            )
        end = ends[0]
        first = start if self.include_markers else start + 1
        last = end + 1 if self.include_markers else end
        return "".join(lines[first:last])


def select_resource(
    resource: Resource,
    selector: str | ResourceSelector | None = None,
) -> Any:
    """Select a value, inferring string selectors only for structured resources."""
    if selector is None:
        return resource.data
    if isinstance(selector, str):
        if resource.media_type not in _STRUCTURED_MEDIA_TYPES:
            raise ResourceSelectorError(
                "string selectors are JSON Pointers and require a structured JSON, YAML, "
                "or TOML resource; pass an explicit selector object for text"
            )
        return JsonPointer(selector).select(resource)
    return selector.select(resource)


__all__ = [
    "IdentitySelector",
    "JsonPointer",
    "LineRange",
    "MarkdownHeading",
    "NamedBlock",
    "ResourceSelector",
    "select_resource",
]
