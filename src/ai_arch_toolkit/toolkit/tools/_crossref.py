"""Crossref tools — public DOI and scholarly metadata lookup."""

from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from typing import Any

from ai_arch_toolkit.core import tool

_API_URL = "https://api.crossref.org/works"
_TIMEOUT = 10
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_MAX_RESULTS_LIMIT = 20
_ABSTRACT_MAX_CHARS = 900
_VALID_TYPE_FILTER = re.compile(r"^[a-z0-9-]+$")
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True, kw_only=True)
class _CrossrefWork:
    """Normalized metadata for a Crossref work."""

    doi: str
    title: str
    authors: tuple[str, ...]
    published: str
    container_title: str
    publisher: str
    work_type: str
    url: str
    abstract: str
    referenced_by_count: int | None
    references: tuple[str, ...]
    license_urls: tuple[str, ...]
    links: tuple[str, ...]


@tool
def crossref_search(
    query: str,
    max_results: int = 5,
    start: int = 0,
    from_date: str = "",
    to_date: str = "",
    type_filter: str = "",
) -> str:
    """Search Crossref works using the public Crossref REST API.

    Args:
        query: Search text, such as a paper title, topic, author, DOI, or citation fragment.
        max_results: Number of works to return (1-20). Defaults to 5.
        start: Zero-based result offset for pagination. Defaults to 0.
        from_date: Optional publication date lower bound as YYYY-MM-DD.
        to_date: Optional publication date upper bound as YYYY-MM-DD.
        type_filter: Optional Crossref type, e.g. "journal-article" or "proceedings-article".
    """
    query = query.strip()
    if not query:
        return "Crossref search failed: query cannot be empty."
    if start < 0:
        return "Crossref search failed: start must be greater than or equal to 0."

    max_results = max(1, min(max_results, _MAX_RESULTS_LIMIT))
    filter_value = _build_filter(from_date, to_date, type_filter)
    if filter_value.startswith("Crossref search failed:"):
        return filter_value

    params = {
        "query": query,
        "rows": str(max_results),
        "offset": str(start),
    }
    if filter_value:
        params["filter"] = filter_value

    try:
        data = _fetch_crossref("", params)
        items = data.get("message", {}).get("items", [])
        works = [_parse_work(item) for item in items if isinstance(item, dict)]
    except urllib.error.HTTPError as e:
        return f"Crossref search failed: HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"Crossref search failed: URL error: {e.reason}"
    except TimeoutError:
        return "Crossref search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"Crossref search failed: could not parse API response: {e}"

    if not works:
        return f"No Crossref results for: {query!r}"

    return f"Crossref results for {query!r}:\n" + _format_works(works, include_abstract=False)


@tool
def crossref_work(doi: str) -> str:
    """Fetch Crossref metadata for a specific DOI.

    Args:
        doi: DOI string or DOI URL, e.g. "10.1038/nature14539" or "https://doi.org/...".
    """
    normalized = _normalize_doi(doi)
    if not normalized:
        return f"Crossref work lookup failed: invalid DOI: {doi!r}"

    try:
        data = _fetch_crossref(f"/{urllib.parse.quote(normalized, safe='')}", {})
        message = data.get("message", {})
        if not isinstance(message, dict):
            return f"Crossref work not found: {normalized}"
        work = _parse_work(message)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"Crossref work not found: {normalized}"
        return f"Crossref work lookup failed: HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"Crossref work lookup failed: URL error: {e.reason}"
    except TimeoutError:
        return "Crossref work lookup failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"Crossref work lookup failed: could not parse API response: {e}"

    return f"Crossref work {normalized}:\n" + _format_works(
        [work],
        include_index=False,
        include_abstract=True,
        include_references=True,
    )


def _build_filter(from_date: str, to_date: str, type_filter: str) -> str:
    filters: list[str] = []
    from_date = from_date.strip()
    to_date = to_date.strip()
    type_filter = type_filter.strip()

    parsed_start: date | None = None
    parsed_end: date | None = None
    if from_date:
        parsed_start = _parse_date(from_date)
        if parsed_start is None:
            return f"Crossref search failed: invalid from_date {from_date!r}. Use YYYY-MM-DD."
        filters.append(f"from-pub-date:{from_date}")
    if to_date:
        parsed_end = _parse_date(to_date)
        if parsed_end is None:
            return f"Crossref search failed: invalid to_date {to_date!r}. Use YYYY-MM-DD."
        filters.append(f"until-pub-date:{to_date}")
    if parsed_start and parsed_end and parsed_start > parsed_end:
        return "Crossref search failed: from_date must be before or equal to to_date."
    if type_filter:
        if not _VALID_TYPE_FILTER.fullmatch(type_filter):
            return f"Crossref search failed: invalid type_filter: {type_filter!r}"
        filters.append(f"type:{type_filter}")
    return ",".join(filters)


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _fetch_crossref(path: str, params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{_API_URL}{path}"
    if query:
        url = f"{url}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read())


def _parse_work(item: dict[str, Any]) -> _CrossrefWork:
    return _CrossrefWork(
        doi=str(item.get("DOI", "")).strip(),
        title=_title(item),
        authors=_authors(item),
        published=_published_date(item),
        container_title=_first_string(item.get("container-title")),
        publisher=str(item.get("publisher", "")).strip(),
        work_type=str(item.get("type", "")).strip(),
        url=str(item.get("URL", "")).strip(),
        abstract=_clean_text(str(item.get("abstract", "")).strip()),
        referenced_by_count=_int_or_none(item.get("is-referenced-by-count")),
        references=_references(item),
        license_urls=_license_urls(item),
        links=_links(item),
    )


def _title(item: dict[str, Any]) -> str:
    title = _first_string(item.get("title"))
    subtitle = _first_string(item.get("subtitle"))
    if title and subtitle:
        return f"{title}: {subtitle}"
    return title or "(untitled)"


def _authors(item: dict[str, Any]) -> tuple[str, ...]:
    authors: list[str] = []
    for author in item.get("author", []):
        if not isinstance(author, dict):
            continue
        name = str(author.get("name", "")).strip()
        if not name:
            given = str(author.get("given", "")).strip()
            family = str(author.get("family", "")).strip()
            name = " ".join(part for part in (given, family) if part)
        if name:
            authors.append(name)
    return tuple(authors)


def _published_date(item: dict[str, Any]) -> str:
    for key in ("published-print", "published-online", "published", "issued", "created"):
        value = item.get(key)
        if isinstance(value, dict):
            date_parts = value.get("date-parts")
            if isinstance(date_parts, list) and date_parts:
                return _format_date_parts(date_parts[0])
    return ""


def _format_date_parts(parts: Any) -> str:
    if not isinstance(parts, list) or not parts:
        return ""
    values = [str(part).zfill(2) for part in parts[:3]]
    if values:
        values[0] = values[0].lstrip("0") or "0"
    return "-".join(values)


def _references(item: dict[str, Any]) -> tuple[str, ...]:
    references: list[str] = []
    for ref in item.get("reference", []):
        if not isinstance(ref, dict):
            continue
        parts = [
            str(ref.get("author", "")).strip(),
            str(ref.get("article-title", "")).strip(),
            str(ref.get("journal-title", "")).strip(),
            str(ref.get("year", "")).strip(),
            str(ref.get("DOI", "")).strip(),
        ]
        text = "; ".join(part for part in parts if part)
        if not text:
            text = str(ref.get("unstructured", "")).strip()
        if text:
            references.append(_clean_text(text))
    return tuple(references)


def _license_urls(item: dict[str, Any]) -> tuple[str, ...]:
    urls: list[str] = []
    for license_item in item.get("license", []):
        if isinstance(license_item, dict):
            url = str(license_item.get("URL", "")).strip()
            if url:
                urls.append(url)
    return tuple(dict.fromkeys(urls))


def _links(item: dict[str, Any]) -> tuple[str, ...]:
    urls: list[str] = []
    for link in item.get("link", []):
        if isinstance(link, dict):
            url = str(link.get("URL", "")).strip()
            if url:
                urls.append(url)
    return tuple(dict.fromkeys(urls))


def _normalize_doi(doi: str) -> str:
    value = doi.strip()
    if not value:
        return ""
    lower = value.lower()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ):
        if lower.startswith(prefix):
            value = value[len(prefix) :]
            break
    if value.lower().startswith("doi:"):
        value = value[4:]
    value = value.strip()
    if " " in value or not value.lower().startswith("10.") or "/" not in value:
        return ""
    return value


def _format_works(
    works: list[_CrossrefWork],
    *,
    include_index: bool = True,
    include_abstract: bool = False,
    include_references: bool = False,
) -> str:
    blocks: list[str] = []
    for index, work in enumerate(works, start=1):
        title = f"{index}. {work.title}" if include_index else work.title
        lines = [title]

        meta: list[str] = []
        if work.doi:
            meta.append(f"DOI: {work.doi}")
        if work.work_type:
            meta.append(f"type: {work.work_type}")
        if work.published:
            meta.append(f"published: {work.published}")
        if meta:
            lines.append("   " + " | ".join(meta))

        if work.authors:
            lines.append(f"   Authors: {_format_authors(work.authors)}")
        if work.container_title:
            lines.append(f"   Venue: {work.container_title}")
        if work.publisher:
            lines.append(f"   Publisher: {work.publisher}")
        if work.referenced_by_count is not None:
            lines.append(f"   Referenced by: {work.referenced_by_count}")
        if include_abstract and work.abstract:
            lines.append(f"   Abstract: {_truncate(work.abstract, _ABSTRACT_MAX_CHARS)}")
        if work.url:
            lines.append(f"   URL: {work.url}")
        if work.license_urls:
            lines.append("   License: " + " | ".join(work.license_urls[:3]))
        if work.links:
            lines.append("   Links: " + " | ".join(work.links[:3]))
        if include_references and work.references:
            lines.append(f"   References ({len(work.references)} deposited):")
            for reference in work.references[:5]:
                lines.append(f"     - {_truncate(reference, 220)}")
            if len(work.references) > 5:
                lines.append(f"     ... (+{len(work.references) - 5} more)")

        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_authors(authors: tuple[str, ...]) -> str:
    if len(authors) <= 8:
        return ", ".join(authors)
    return ", ".join(authors[:8]) + f", ... (+{len(authors) - 8} more)"


def _first_string(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return _clean_text(item)
    if isinstance(value, str):
        return _clean_text(value)
    return ""


def _clean_text(text: str) -> str:
    return " ".join(html.unescape(_TAG_RE.sub(" ", text)).split())


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ... [truncated]"
