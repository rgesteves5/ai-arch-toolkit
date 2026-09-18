"""MediaWiki and Wiktionary tools — public wiki search and page parsing."""

from __future__ import annotations

import html
import re
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.toolkit.tools._http import Api, HttpError

_DEFAULT_API = "https://en.wiktionary.org/w/api.php"
_TIMEOUT_S = 15
_STATUS_MESSAGES = {404: "no matching records found."}
_WIKTIONARY = Api(
    base=_DEFAULT_API, name="MediaWiki", timeout_s=_TIMEOUT_S, status_messages=_STATUS_MESSAGES
)
_MAX_LIMIT = 25
_TEXT_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,180}$", re.UNICODE)
_LANG_RE = re.compile(r"^[A-Za-z -]{1,80}$")
# An api_url from the model must be on one of these hosts, or a subdomain of one.
_DOMAINS = frozenset(
    {
        "wikipedia.org",
        "wikimedia.org",
        "wiktionary.org",
        "wikidata.org",
        "wikibooks.org",
        "wikiquote.org",
        "wikisource.org",
        "wikiversity.org",
        "wikivoyage.org",
        "wikinews.org",
        "mediawiki.org",
    }
)


@tool(capability="network")
def mediawiki_search(
    query: str,
    api_url: str = _DEFAULT_API,
    max_results: int = 10,
    offset: int = 0,
) -> str:
    """Search a MediaWiki API.

    Args:
        query: Search text.
        api_url: MediaWiki API endpoint. Defaults to English Wiktionary.
        max_results: Number of pages to return (1-25). Defaults to 10.
        offset: Zero-based result offset. Defaults to 0.
    """
    if not _valid_text(query):
        return "MediaWiki search failed: invalid query."
    api = _api(api_url)
    if api is None:
        return "MediaWiki search failed: invalid api_url."
    if offset < 0:
        return "MediaWiki search failed: offset must be greater than or equal to 0."
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query.strip(),
        "srlimit": str(_bounded(max_results)),
        "sroffset": str(offset),
        "format": "json",
        "utf8": "1",
    }
    try:
        return api.get_json(params=params, parse=lambda data: _search_text(data, query, offset))
    except HttpError as e:
        return f"MediaWiki search failed: {e}"


@tool(capability="network")
def mediawiki_page(title: str, api_url: str = _DEFAULT_API, max_chars: int = 1200) -> str:
    """Fetch and lightly clean a MediaWiki page's wikitext.

    Args:
        title: Page title.
        api_url: MediaWiki API endpoint. Defaults to English Wiktionary.
        max_chars: Maximum cleaned characters to return (200-4000). Defaults to 1200.
    """
    if not _valid_text(title):
        return "MediaWiki page failed: invalid title."
    api = _api(api_url)
    if api is None:
        return "MediaWiki page failed: invalid api_url."
    try:
        return api.get_json(
            params=_parse_params(title.strip(), "wikitext|sections"),
            parse=lambda data: _page_text(data, title, max_chars),
        )
    except HttpError as e:
        return f"MediaWiki page failed: {e}"


@tool(capability="network")
def mediawiki_sections(title: str, api_url: str = _DEFAULT_API) -> str:
    """List sections for a MediaWiki page.

    Args:
        title: Page title.
        api_url: MediaWiki API endpoint. Defaults to English Wiktionary.
    """
    if not _valid_text(title):
        return "MediaWiki sections failed: invalid title."
    api = _api(api_url)
    if api is None:
        return "MediaWiki sections failed: invalid api_url."
    try:
        return api.get_json(
            params=_parse_params(title.strip(), "sections"),
            parse=lambda data: _sections_text(data, title),
        )
    except HttpError as e:
        return f"MediaWiki sections failed: {e}"


@tool(capability="network")
def wiktionary_entry(term: str, language: str = "English", max_chars: int = 1600) -> str:
    """Fetch a Wiktionary entry and focus on one language section.

    Args:
        term: Wiktionary term/page title.
        language: Language section to prioritize. Defaults to English.
        max_chars: Maximum cleaned characters to return (200-4000). Defaults to 1600.
    """
    if not _valid_text(term):
        return "Wiktionary entry failed: invalid term."
    if not _LANG_RE.fullmatch(language.strip()):
        return "Wiktionary entry failed: invalid language."
    try:
        return _WIKTIONARY.get_json(
            params=_parse_params(term.strip(), "wikitext|sections"),
            parse=lambda data: _entry_text(data, term, language.strip(), max_chars),
        )
    except HttpError as e:
        return f"Wiktionary entry failed: {e}"


def _api(api_url: str) -> Api | None:
    """The MediaWiki API at ``api_url``; ``None`` unless it is a Wikimedia ``https://…/api.php``."""
    try:
        api = Api.within(
            api_url.strip(),
            _DOMAINS,
            name="MediaWiki",
            timeout_s=_TIMEOUT_S,
            status_messages=_STATUS_MESSAGES,
        )
    except HttpError:
        return None
    return api if api.base.endswith("api.php") else None


def _parse_params(title: str, props: str) -> dict[str, str]:
    return {"action": "parse", "page": title, "prop": props, "format": "json", "utf8": "1"}


def _search_text(data: dict[str, Any], query: str, offset: int) -> str:
    items = data.get("query", {}).get("search", [])
    if not isinstance(items, list) or not items:
        return "No MediaWiki pages found."
    total = _string(data.get("query", {}).get("searchinfo", {}).get("totalhits")) or "?"
    lines = [
        f"MediaWiki pages for {query!r} (returned {len(items)}, total {total}, offset {offset}):"
    ]
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        snippet = _strip_html(_string(item.get("snippet")))
        lines.append(
            f"{index}. {_string(item.get('title'))} | pageid: {_string(item.get('pageid'))}"
        )
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


def _page_text(data: dict[str, Any], title: str, max_chars: int) -> str:
    parse = data.get("parse", {})
    if not isinstance(parse, dict):
        return f"MediaWiki page not found: {title}"
    page_title = _string(parse.get("title")) or title.strip()
    text = _extract_wikitext(parse)
    cleaned = _clean_wikitext(text)
    sections = _section_titles(parse)
    limit = max(200, min(max_chars, 4000))
    lines = [f"MediaWiki page {page_title}:"]
    if sections:
        lines.append("   sections: " + "; ".join(sections[:15]))
    if cleaned:
        lines.append(_trim(cleaned, limit))
    return "\n".join(lines)


def _sections_text(data: dict[str, Any], title: str) -> str:
    parse = data.get("parse", {})
    if not isinstance(parse, dict):
        return f"MediaWiki page not found: {title}"
    sections = parse.get("sections", [])
    if not isinstance(sections, list) or not sections:
        return f"No MediaWiki sections found for {title}."
    lines = [f"MediaWiki sections for {_string(parse.get('title')) or title.strip()}:"]
    for section in sections[:_MAX_LIMIT]:
        if isinstance(section, dict):
            lines.append(
                f"{_string(section.get('index'))}. {_string(section.get('line'))} "
                f"| level: {_string(section.get('level'))}"
            )
    return "\n".join(lines)


def _entry_text(data: dict[str, Any], term: str, language: str, max_chars: int) -> str:
    parse = data.get("parse", {})
    if not isinstance(parse, dict):
        return f"Wiktionary entry not found: {term}"
    text = _extract_wikitext(parse)
    focused = _language_section(text, language) or text
    cleaned = _clean_wikitext(focused)
    sections = _section_titles(parse)
    limit = max(200, min(max_chars, 4000))
    lines = [f"Wiktionary entry {term.strip()} ({language}):"]
    if sections:
        lines.append("   available sections: " + "; ".join(sections[:20]))
    if cleaned:
        lines.append(_trim(cleaned, limit))
    return "\n".join(lines)


def _extract_wikitext(parse: dict[str, Any]) -> str:
    value = parse.get("wikitext", {})
    if isinstance(value, dict):
        raw = value.get("*")
        return raw if isinstance(raw, str) else _string(raw)
    return value if isinstance(value, str) else _string(value)


def _section_titles(parse: dict[str, Any]) -> list[str]:
    sections = parse.get("sections", [])
    if not isinstance(sections, list):
        return []
    return [_string(section.get("line")) for section in sections if isinstance(section, dict)]


def _language_section(text: str, language: str) -> str:
    pattern = re.compile(rf"^==\s*{re.escape(language)}\s*==\s*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return ""
    next_lang = re.search(r"^==[^=].*==\s*$", text[match.end() :], re.MULTILINE)
    end = match.end() + next_lang.start() if next_lang else len(text)
    return text[match.end() : end]


def _clean_wikitext(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"\{\{[^{}]*\}\}", "", cleaned)
    cleaned = re.sub(r"<ref[^>]*>.*?</ref>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", cleaned)
    cleaned = re.sub(r"\[\[([^\]]+)\]\]", r"\1", cleaned)
    cleaned = re.sub(r"'{2,5}", "", cleaned)
    cleaned = re.sub(r"^=+\s*(.*?)\s*=+$", r"\1:", cleaned, flags=re.MULTILINE)
    cleaned = html.unescape(cleaned)
    return "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())


def _strip_html(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value))


def _valid_text(value: str) -> bool:
    return bool(_TEXT_RE.fullmatch(value.strip()))


def _bounded(value: int) -> int:
    return max(1, min(value, _MAX_LIMIT))


def _trim(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
