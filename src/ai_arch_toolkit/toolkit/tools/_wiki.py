"""The wiki family: search, outline and read the pages of any Wikimedia wiki, and Wiktionary
entries (T05; D39 to D41).

A page is read whole through ``action=parse``, as the HTML the wiki renders, and converted to
text by ``_wiki_html`` (D40). Its sections are the headings of that text, numbered in order: a
section is cut from the page's text, never asked of the wiki, so ``wiki_outline``'s sizes are
what ``wiki_read`` returns, a section a template brings in is a section too, and every request is
one the wiki serves from its cache (a ``section=`` request is parsed afresh;
https://www.mediawiki.org/wiki/API:Parsing_wikitext/TOCData). Every long answer goes through the
window (D39): its footer says what was shown and the call that reads on. English Wikipedia is the
default wiki; any Wikimedia wiki is one argument away (``wiki="en.wikibooks.org"``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._http import HttpError
from ai_arch_toolkit.toolkit.tools._mediawiki import (
    WIKIMEDIA_DOMAINS,
    WIKIPEDIA,
    WIKTIONARY,
    wiki_api,
    wiki_host,
)
from ai_arch_toolkit.toolkit.tools._wiki_html import Heading, WikiText, wiki_text
from ai_arch_toolkit.toolkit.tools._window import (
    Window,
    find_window,
    list_window,
    page_window,
    text_window,
)

# The hosts these tools' requests may reach (the tools' invariants read it).
_DOMAINS = WIKIMEDIA_DOMAINS
# CirrusSearch serves results up to the 10,000th (https://www.mediawiki.org/wiki/API:Search).
_SEARCH_DEPTH = 10_000
_OUTLINE_PAGE = 50
# The longest title MediaWiki takes (https://www.mediawiki.org/wiki/Manual:Page_title).
_TITLE_BYTES = 255
_MAX_CHARS = 20_000
_DEFAULT_CHARS = 6_000


@tool(capability="network")
def wiki_search(
    query: str,
    wiki: str = WIKIPEDIA,
    max_results: Annotated[int, Range(1, 50)] = 10,
    offset: Annotated[int, Range(0, _SEARCH_DEPTH - 1)] = 0,
) -> ToolResult:
    """Search the pages of a Wikimedia wiki, English Wikipedia by default.

    The query takes the wiki's search syntax: words, "an exact phrase", ``intitle:word``,
    ``incategory:Name``, or ``morelike:Title`` for the pages most like that one.

    Args:
        query: What to look for.
        wiki: The wiki's host, e.g. "pt.wikipedia.org", "en.wikibooks.org" or
            "en.wiktionary.org".
        max_results: How many pages to list.
        offset: How many results to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when ``query`` is empty or ``wiki`` is not a Wikimedia wiki.
    """
    if not query.strip():
        raise ToolFailure("validation_error", "query cannot be empty; say what to search for")
    host = wiki_host(wiki)
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query.strip(),
        "srlimit": str(min(max_results, _SEARCH_DEPTH - offset)),
        "sroffset": str(offset),
        "srinfo": "totalhits|suggestion",
        "srprop": "snippet|size|wordcount",
        "format": "json",
        "formatversion": "2",
    }
    return wiki_api(host).get_json(
        params=params, parse=lambda data: _search_answer(data, query.strip(), host, offset)
    )


@tool(capability="network")
def wiki_outline(
    title: str, wiki: str = WIKIPEDIA, offset: Annotated[int, Range(0)] = 0
) -> ToolResult:
    """List a wiki page's sections: the number ``wiki_read`` takes, the heading and its size.

    Section 0 is the introduction; a section's size counts its subsections, as ``wiki_read``
    returns them.

    Args:
        title: The page's title, as the wiki's search returns it.
        wiki: The wiki's host, e.g. "en.wikibooks.org".
        offset: How many sections to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when ``title`` or ``wiki`` is invalid; not_found when the
            wiki has no page with that title.
    """
    return _page(wiki_host(wiki), title, lambda page: _outline_answer(page, offset))


@tool(capability="network")
def wiki_read(
    title: str,
    wiki: str = WIKIPEDIA,
    section: Annotated[int, Range(0)] | None = None,
    find: str = "",
    offset: Annotated[int, Range(0)] = 0,
    max_chars: Annotated[int, Range(500, _MAX_CHARS)] = _DEFAULT_CHARS,
) -> ToolResult:
    """Read a wiki page as text: whole, one section, or the passages that mention a term.

    Tables come one row per line, with every cell, so a row found by a term reads whole.

    Args:
        title: The page's title, as the wiki's search returns it.
        wiki: The wiki's host, e.g. "en.wikibooks.org".
        section: The section's number from ``wiki_outline`` (0 is the introduction); none for
            the whole page.
        find: A term to look for: the answer is the passages around each match, not the page.
        offset: Where to start, in characters of the page or the section (with ``find``, where
            to search on from); the footer gives the next offset.
        max_chars: How many characters to return.

    Raises:
        ToolFailure: validation_error when ``title`` or ``wiki`` is invalid, or the page has no
            such section; not_found when the wiki has no page with that title.
    """
    term = find.strip()

    def read(page: _Page) -> ToolResult:
        where, text = page.where, page.text.text
        if section is not None:
            part = _section(page, section)
            where, text = f"{where}, section {section} ({part.title})", part.text
        if not text.strip():
            return ToolResult.success(f"{where} has no text.")
        if term:
            window = find_window(text, term, offset=offset, limit=max_chars)
            heading = f"{where}, passages that mention {term!r}:"
        else:
            window = text_window(text, offset=offset, limit=max_chars)
            heading = f"{where}:"
        return _within_section(window, section).result(heading=heading)

    return _page(wiki_host(wiki), title, read)


@tool(capability="network")
def wiktionary_entry(
    term: str,
    language: str = "English",
    offset: Annotated[int, Range(0)] = 0,
    max_chars: Annotated[int, Range(500, _MAX_CHARS)] = _DEFAULT_CHARS,
) -> ToolResult:
    """Read the English Wiktionary's entry for a term in one language: its senses by part of
    speech, with pronunciation, etymology and examples.

    Args:
        term: The word or phrase, as Wiktionary titles it (case matters: "Polish" and "polish"
            are two entries).
        language: The language's English name, as Wiktionary heads its section, e.g. "French".
        offset: Where to start, in characters; the footer gives the next offset.
        max_chars: How many characters to return.

    Raises:
        ToolFailure: validation_error when ``term`` or ``language`` is invalid; not_found when
            Wiktionary has no entry for the term, or none in that language.
    """
    wanted = " ".join(language.split())
    if not wanted or len(wanted) > 80 or any(mark in wanted for mark in "<>[]{}|#"):
        raise ToolFailure(
            "validation_error",
            f"invalid language {language[:100]!r}; give its English name, e.g. 'French'",
        )

    def read(page: _Page) -> ToolResult:
        text = _language_text(page, wanted)
        heading = f"Wiktionary, {page.title} ({wanted}):"
        return text_window(text, offset=offset, limit=max_chars).result(heading=heading)

    return _page(WIKTIONARY, term, read)


# --- The page ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class _Page:
    """A page as the tools read it: its title, where it was read (and on which wiki), and its
    text."""

    title: str
    where: str
    host: str
    text: WikiText


@dataclass(frozen=True, slots=True)
class _Part:
    """A section of a page: its heading's text and its own text, subsections included."""

    title: str
    text: str


def _page[T](host: str, title: str, read: Callable[[_Page], T]) -> T:
    """``title`` on ``host`` (redirects followed), read by ``read`` inside the door: a shape
    ``read`` did not expect is the API's failure.

    Raises:
        ToolFailure: validation_error for an empty or too long title (or one the wiki refuses);
            not_found when the wiki has no page with that title, with the next step.
    """
    name = title.strip()
    if not name or len(name.encode()) > _TITLE_BYTES:
        raise ToolFailure(
            "validation_error",
            f"invalid title {title[:100]!r}; give a page title of 1 to {_TITLE_BYTES} bytes, "
            "as the wiki's search returns it",
        )
    params = {
        "action": "parse",
        "page": name,
        "prop": "text",
        "redirects": "1",
        "disableeditsection": "1",
        "disabletoc": "1",
        "disablelimitreport": "1",
        "format": "json",
        "formatversion": "2",
    }
    try:
        return wiki_api(host).get_json(
            params=params, parse=lambda data: read(_page_of(data, host))
        )
    except HttpError as failure:  # what the wiki reported, not what ``read`` found
        if failure.error.type != "not_found":  # missingtitle is the only not_found it reports
            raise
        raise ToolFailure("not_found", _missing(host, name)) from failure


def _page_of(data: dict[str, Any], host: str) -> _Page:
    """The page of an ``action=parse`` answer; an error answer never gets here
    (``mediawiki_error``)."""
    parse = data.get("parse")
    if not isinstance(parse, dict):
        # The wiki sends a page over its size limit (12 MiB) as a warning, without the page.
        said = _string(data.get("warnings"))[:300]
        raise ToolFailure(
            "upstream",
            'could not parse API response: no "parse" object'
            + (f" (the wiki warned: {said})" if said else ""),
        )
    title = _string(parse.get("title"))
    redirects = [r for r in parse.get("redirects", []) if isinstance(r, dict)]
    came = f", redirected from {_string(redirects[0].get('from'))!r}" if redirects else ""
    html = parse.get("text")
    return _Page(
        title=title,
        where=f"{title} ({host}{came})",
        host=host,
        text=wiki_text(html if isinstance(html, str) else ""),
    )


def _missing(host: str, title: str) -> str:
    search = "wiki_search" if host == WIKIPEDIA else f"wiki_search(wiki={host!r})"
    return f"{host} has no page titled {title!r}; find the exact title with {search}"


def _section(page: _Page, number: int) -> _Part:
    """Section ``number`` of ``page``: 0 is the text before the first heading; ``N`` the N-th
    heading's, up to the next heading of its level or above.

    Raises:
        ToolFailure: validation_error when the page has no such section.
    """
    headings, text = page.text.headings, page.text.text
    if number == 0:
        return _Part("introduction", text[: headings[0].start if headings else len(text)])
    if number > len(headings):
        raise ToolFailure(
            "validation_error",
            f"{page.title} has {len(headings)} sections, not {number}; list them with "
            "wiki_outline",
        )
    heading = headings[number - 1]
    return _Part(heading.title, text[heading.start : _ends(page.text)[number - 1]])


def _ends(converted: WikiText) -> list[int]:
    """Where each heading's section ends: at the next heading of its level or above, or at the
    end of the text (one pass, whatever the depth)."""
    ends = [len(converted.text)] * len(converted.headings)
    open_sections: list[tuple[int, Heading]] = []
    for index, heading in enumerate(converted.headings):
        while open_sections and open_sections[-1][1].level >= heading.level:
            ends[open_sections.pop()[0]] = heading.start
        open_sections.append((index, heading))
    return ends


def _within_section(window: Window, section: int | None) -> Window:
    """A window of one section names the section in its next call: its offsets are the
    section's."""
    if section is None or window.next_call is None:
        return window
    return replace(window, next_call={**window.next_call, "section": section})


# --- Answers -----------------------------------------------------------------------------------


def _search_answer(data: dict[str, Any], query: str, host: str, offset: int) -> ToolResult:
    found = data.get("query", {})
    items = [item for item in found.get("search", []) if isinstance(item, dict)]
    info = found.get("searchinfo", {}) if isinstance(found.get("searchinfo"), dict) else {}
    total = info.get("totalhits") if isinstance(info.get("totalhits"), int) else None
    if not items and offset == 0:
        suggestion = _string(info.get("suggestion"))
        hint = f" Did you mean {suggestion!r}?" if suggestion else ""
        return ToolResult.success(f"No pages on {host} match {query!r}.{hint}")
    lines = [_search_line(offset + number, item) for number, item in enumerate(items, start=1)]
    onward = data.get("continue", {})
    next_offset = onward.get("sroffset") if isinstance(onward, dict) else None
    next_call = {"offset": next_offset} if isinstance(next_offset, int) else None
    window = list_window(lines, first=offset + 1, total=total, next_call=next_call)
    return window.result(heading=f"Pages on {host} that match {query!r}:")


def _search_line(number: int, item: dict[str, Any]) -> str:
    snippet = wiki_text(_string(item.get("snippet"))).text.replace("\n", " ")
    words = item.get("wordcount")
    size = f" ({words} words)" if isinstance(words, int) else ""
    return f"{number}. {_string(item.get('title'))}{size}" + (f": {snippet}" if snippet else "")


def _outline_answer(page: _Page, offset: int) -> ToolResult:
    headings = page.text.headings
    top = min((h.level for h in headings), default=1)
    lines = [f"0. (introduction): {len(_section(page, 0).text)} chars"]
    for number, (heading, end) in enumerate(zip(headings, _ends(page.text), strict=True), 1):
        indent = "  " * min(heading.level - top, 5)
        lines.append(f"{indent}{number}. {heading.title}: {end - heading.start} chars")
    window = page_window(lines, offset=offset, limit=_OUTLINE_PAGE)
    on = "" if page.host == WIKIPEDIA else f", wiki={page.host!r}"
    return window.result(
        heading=f"{page.where}, sections (read one with wiki_read(section=N{on})):"
    )


def _language_text(page: _Page, language: str) -> str:
    """The text of ``language``'s section of a Wiktionary entry (a level-2 heading), up to the
    next language's (https://en.wiktionary.org/wiki/Wiktionary:Entry_layout).

    Raises:
        ToolFailure: not_found when the entry has no section for ``language``, with those it has.
    """
    headings = page.text.headings
    languages = [(index, h) for index, h in enumerate(headings) if h.level == 2]
    for index, heading in languages:
        if heading.title.casefold() == language.casefold():
            return page.text.text[heading.start : _ends(page.text)[index]]
    has = ", ".join(heading.title for _, heading in languages)
    raise ToolFailure(
        "not_found",
        f"Wiktionary's entry {page.title!r} has no {language} section"
        + (f"; it has {has}" if has else ""),
    )


def _string(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
