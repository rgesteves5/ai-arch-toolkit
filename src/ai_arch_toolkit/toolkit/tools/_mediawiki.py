"""MediaWiki and Wiktionary tools — public wiki search and page parsing."""

from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ai_arch_toolkit.core import tool

_DEFAULT_API = "https://en.wiktionary.org/w/api.php"
_TIMEOUT = 15
_USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"
_MAX_LIMIT = 25
_TEXT_RE = re.compile(r"^[\w\s,.'()/%:+-]{1,180}$", re.UNICODE)
_LANG_RE = re.compile(r"^[A-Za-z -]{1,80}$")


@tool
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
    if not _valid_api_url(api_url):
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
        data = _fetch_json(api_url, params)
        items = data.get("query", {}).get("search", [])
    except urllib.error.HTTPError as e:
        return _http_error("MediaWiki search failed", e)
    except urllib.error.URLError as e:
        return f"MediaWiki search failed: URL error: {e.reason}"
    except TimeoutError:
        return "MediaWiki search failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"MediaWiki search failed: could not parse API response: {e}"

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


@tool
def mediawiki_page(title: str, api_url: str = _DEFAULT_API, max_chars: int = 1200) -> str:
    """Fetch and lightly clean a MediaWiki page's wikitext.

    Args:
        title: Page title.
        api_url: MediaWiki API endpoint. Defaults to English Wiktionary.
        max_chars: Maximum cleaned characters to return (200-4000). Defaults to 1200.
    """
    if not _valid_text(title):
        return "MediaWiki page failed: invalid title."
    if not _valid_api_url(api_url):
        return "MediaWiki page failed: invalid api_url."
    try:
        data = _parse_page(api_url, title.strip(), props="wikitext|sections")
    except urllib.error.HTTPError as e:
        return _http_error("MediaWiki page failed", e)
    except urllib.error.URLError as e:
        return f"MediaWiki page failed: URL error: {e.reason}"
    except TimeoutError:
        return "MediaWiki page failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"MediaWiki page failed: could not parse API response: {e}"

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


@tool
def mediawiki_sections(title: str, api_url: str = _DEFAULT_API) -> str:
    """List sections for a MediaWiki page.

    Args:
        title: Page title.
        api_url: MediaWiki API endpoint. Defaults to English Wiktionary.
    """
    if not _valid_text(title):
        return "MediaWiki sections failed: invalid title."
    if not _valid_api_url(api_url):
        return "MediaWiki sections failed: invalid api_url."
    try:
        data = _parse_page(api_url, title.strip(), props="sections")
        parse = data.get("parse", {})
    except urllib.error.HTTPError as e:
        return _http_error("MediaWiki sections failed", e)
    except urllib.error.URLError as e:
        return f"MediaWiki sections failed: URL error: {e.reason}"
    except TimeoutError:
        return "MediaWiki sections failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"MediaWiki sections failed: could not parse API response: {e}"

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


@tool
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
        data = _parse_page(_DEFAULT_API, term.strip(), props="wikitext|sections")
    except urllib.error.HTTPError as e:
        return _http_error("Wiktionary entry failed", e)
    except urllib.error.URLError as e:
        return f"Wiktionary entry failed: URL error: {e.reason}"
    except TimeoutError:
        return "Wiktionary entry failed: request timed out."
    except (json.JSONDecodeError, TypeError) as e:
        return f"Wiktionary entry failed: could not parse API response: {e}"

    parse = data.get("parse", {})
    if not isinstance(parse, dict):
        return f"Wiktionary entry not found: {term}"
    text = _extract_wikitext(parse)
    focused = _language_section(text, language.strip()) or text
    cleaned = _clean_wikitext(focused)
    sections = _section_titles(parse)
    limit = max(200, min(max_chars, 4000))
    lines = [f"Wiktionary entry {term.strip()} ({language.strip()}):"]
    if sections:
        lines.append("   available sections: " + "; ".join(sections[:20]))
    if cleaned:
        lines.append(_trim(cleaned, limit))
    return "\n".join(lines)


def _parse_page(api_url: str, title: str, *, props: str) -> dict[str, Any]:
    return _fetch_json(
        api_url,
        {
            "action": "parse",
            "page": title,
            "prop": props,
            "format": "json",
            "utf8": "1",
        },
    )


def _fetch_json(api_url: str, params: dict[str, str]) -> dict[str, Any]:
    url = f"{api_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


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


def _http_error(prefix: str, error: urllib.error.HTTPError) -> str:
    if error.code == 404:
        return f"{prefix}: no matching records found."
    if error.code == 429:
        return f"{prefix}: rate limited by MediaWiki (HTTP 429). Try again later."
    return f"{prefix}: HTTP error {error.code}: {error.reason}"


def _valid_text(value: str) -> bool:
    return bool(_TEXT_RE.fullmatch(value.strip()))


def _valid_api_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value.strip())
    return parsed.scheme == "https" and parsed.netloc and parsed.path.endswith("api.php")


def _bounded(value: int) -> int:
    return max(1, min(value, _MAX_LIMIT))


def _trim(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def _string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
