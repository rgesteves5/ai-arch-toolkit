"""Internet Archive tools — public item search and metadata lookup."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.toolkit.tools._http import Api, HttpError

_API = Api(base="https://archive.org", name="Internet Archive", timeout_s=15)
_MAX_RESULTS_LIMIT = 20
_DESCRIPTION_MAX_CHARS = 1000
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True, slots=True, kw_only=True)
class _InternetArchiveItem:
    """Normalized Internet Archive item metadata."""

    identifier: str
    title: str
    creator: tuple[str, ...]
    date: str
    mediatype: str
    collection: tuple[str, ...]
    subjects: tuple[str, ...]
    downloads: int | None
    item_size: int | None
    description: str
    files: tuple[str, ...]


@tool(capability="network")
def internet_archive_search(
    query: str,
    max_results: int = 5,
    page: int = 1,
    mediatype: str = "",
    collection: str = "",
) -> str:
    """Search Internet Archive items using the public advanced search API.

    Args:
        query: Search text.
        max_results: Number of items to return (1-20). Defaults to 5.
        page: One-based result page. Defaults to 1.
        mediatype: Optional mediatype filter, e.g. texts, audio, movies, software.
        collection: Optional collection filter.
    """
    query = query.strip()
    if not query:
        return "Internet Archive search failed: query cannot be empty."
    if page < 1:
        return "Internet Archive search failed: page must be greater than or equal to 1."

    filters = []
    if mediatype.strip():
        filters.append(f"mediatype:{mediatype.strip()}")
    if collection.strip():
        filters.append(f"collection:{collection.strip()}")
    search_query = f"({query})"
    if filters:
        search_query = f"{search_query} AND {' AND '.join(filters)}"

    params = {
        "q": search_query,
        "fl[]": [
            "identifier",
            "title",
            "creator",
            "date",
            "mediatype",
            "collection",
            "subject",
            "downloads",
            "item_size",
        ],
        "rows": str(max(1, min(max_results, _MAX_RESULTS_LIMIT))),
        "page": str(page),
        "output": "json",
    }

    try:
        return _API.get_json(
            "advancedsearch.php", params=params, parse=lambda data: _search_text(data, query)
        )
    except HttpError as e:
        return f"Internet Archive search failed: {e}"


@tool(capability="network")
def internet_archive_item(identifier: str) -> str:
    """Fetch Internet Archive metadata for a specific item identifier.

    Args:
        identifier: Internet Archive item identifier.
    """
    normalized = identifier.strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        return f"Internet Archive item lookup failed: invalid identifier: {identifier!r}"

    try:
        item = _API.get_json("metadata", normalized, parse=_parse_metadata_item)
    except HttpError as e:
        if e.status == 404:
            return f"Internet Archive item not found: {normalized}"
        return f"Internet Archive item lookup failed: {e}"
    if item is None:
        return f"Internet Archive item not found: {normalized}"

    return f"Internet Archive item {normalized}:\n" + _format_items(
        [item],
        include_index=False,
        include_description=True,
    )


def _search_text(data: dict[str, Any], query: str) -> str:
    docs = data.get("response", {}).get("docs", [])
    items = [_parse_search_doc(item) for item in docs if isinstance(item, dict)]
    items = [item for item in items if item is not None]
    if not items:
        return f"No Internet Archive items found for: {query!r}"
    return f"Internet Archive items for {query!r}:\n" + _format_items(items)


def _parse_search_doc(data: dict[str, Any]) -> _InternetArchiveItem | None:
    identifier = _string(data.get("identifier"))
    if not identifier:
        return None
    return _InternetArchiveItem(
        identifier=identifier,
        title=_string(data.get("title")) or "(untitled)",
        creator=_string_tuple(data.get("creator")),
        date=_string(data.get("date")),
        mediatype=_string(data.get("mediatype")),
        collection=_string_tuple(data.get("collection")),
        subjects=_string_tuple(data.get("subject")),
        downloads=_int_or_none(data.get("downloads")),
        item_size=_int_or_none(data.get("item_size")),
        description="",
        files=(),
    )


def _parse_metadata_item(data: dict[str, Any]) -> _InternetArchiveItem | None:
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        return None
    identifier = _string(metadata.get("identifier") or data.get("item"))
    if not identifier:
        return None
    files = []
    for item in data.get("files", []) or []:
        if isinstance(item, dict):
            name = _string(item.get("name"))
            fmt = _string(item.get("format"))
            size = _string(item.get("size"))
            text = name
            if fmt:
                text = f"{text} ({fmt})"
            if size:
                text = f"{text}, {size} bytes"
            if text:
                files.append(text)
    return _InternetArchiveItem(
        identifier=identifier,
        title=_string(metadata.get("title")) or "(untitled)",
        creator=_string_tuple(metadata.get("creator")),
        date=_string(metadata.get("date")),
        mediatype=_string(metadata.get("mediatype")),
        collection=_string_tuple(metadata.get("collection")),
        subjects=_string_tuple(metadata.get("subject")),
        downloads=None,
        item_size=None,
        description=_string(metadata.get("description")),
        files=tuple(files),
    )


def _format_items(
    items: list[_InternetArchiveItem],
    *,
    include_index: bool = True,
    include_description: bool = False,
) -> str:
    blocks: list[str] = []
    for index, item in enumerate(items, start=1):
        title = f"{index}. {item.title}" if include_index else item.title
        lines = [title]
        meta = [f"identifier: {item.identifier}"]
        if item.mediatype:
            meta.append(f"mediatype: {item.mediatype}")
        if item.date:
            meta.append(f"date: {item.date}")
        if item.downloads is not None:
            meta.append(f"downloads: {item.downloads}")
        if item.item_size is not None:
            meta.append(f"size: {item.item_size}")
        lines.append("   " + " | ".join(meta))
        if item.creator:
            lines.append("   Creator: " + ", ".join(item.creator[:8]))
        if item.collection:
            lines.append("   Collections: " + ", ".join(item.collection[:8]))
        if item.subjects:
            lines.append("   Subjects: " + ", ".join(item.subjects[:12]))
        if include_description and item.description:
            lines.append(f"   Description: {_truncate(item.description, _DESCRIPTION_MAX_CHARS)}")
        if item.files:
            lines.append("   Files:")
            for file in item.files[:8]:
                lines.append(f"     - {file}")
        lines.append(f"   URL: https://archive.org/details/{item.identifier}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(_string(item) for item in value if _string(item))
    text = _string(value)
    return (text,) if text else ()


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_string(item) for item in value if _string(item))
    if isinstance(value, dict):
        for key in ("value", "text", "description"):
            text = _string(value.get(key))
            if text:
                return text
        return ""
    return " ".join(str(value).split())


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    try:
        if value is not None and str(value).strip():
            return int(value)
    except ValueError:
        return None
    return None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ... [truncated]"
