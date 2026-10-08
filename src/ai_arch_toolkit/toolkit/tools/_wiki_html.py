"""MediaWiki's rendered HTML as plain text a model reads (D40).

``action=parse`` returns the HTML the wiki renders for a page. Reading it, rather than the
wikitext, gets the templates expanded, the tables whole and the links as their text. The
converter keeps what a reader sees and drops what the wiki marks as navigation, maintenance or
hidden (the classes in ``_NOISE``):

- headings become ``## Title`` lines, with their level, and are listed with their anchor and
  their place in the text, so a page can be read section by section;
- paragraphs, list items (``- `` and ``1. ``, indented by depth) and definition lists keep one
  line each;
- a table becomes one line per row, its cells joined by `` | ``: a cell that spans rows repeats
  on each of them, so every row reads whole, and one that spans columns leaves the others empty;
- a formula reads as its TeX (the ``alttext`` of its MathML, or the ``alt`` of its image), any
  other image as nothing, a figure as its caption.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser

from ai_arch_toolkit.toolkit.tools._tables import Cell, cell, table_lines

# What the wiki marks as not part of the content: edit links, footnote markers and lists, the
# table of contents, navigation boxes, maintenance notices, error text, and everything hidden, not
# printed, or left out of mobile views and excerpts. The core classes are in
# https://www.mediawiki.org/wiki/Manual:Interface/IDs_and_classes; the template ones are the
# names Wikimedia wikis share (https://www.mediawiki.org/wiki/Recommendations_for_mobile_friendly_
# articles_on_Wikimedia_wikis); MediaWiki's own text extraction drops the same (TextExtracts,
# CirrusSearch). Hatnotes stay: they name the page to read next.
_NOISE = frozenset(
    {
        "mw-editsection",
        "reference",
        "references",
        "reflist",
        "mw-references-wrap",
        "mw-cite-backlink",
        "toc",
        "mw-jump-link",
        "noprint",
        "nomobile",
        "noexcerpt",
        "sortkey",
        "error",
        "mw-ext-cite-error",
        "mw-empty-elt",
        "navbox",
        "navbox-styles",
        "vertical-navbox",
        "sidebar",
        "metadata",
        "ambox",
        "ombox",
        "tmbox",
        "cmbox",
        "fmbox",
        "imbox",
        "shortdescription",
        "printfooter",
        "mw-indicators",
        "sr-only",
    }
)
_NOISE_TAGS = frozenset(
    {"style", "script", "link", "meta", "noscript", "template", "audio", "video"}
)
_VOID = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}
)
_BLOCKS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "center",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "header",
        "li",
        "main",
        "ol",
        "p",
        "section",
        "summary",
        "ul",
    }
)
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_TABLE_PARTS = frozenset({"table", "caption", "tr", "td", "th"})
_LIST_PARTS = frozenset({"ul", "ol", "li", "dd"})
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.IGNORECASE)
_MATH_IMAGES = frozenset({"mwe-math-fallback-image-inline", "mwe-math-fallback-image-display"})
_TEX_STYLE = re.compile(r"^\{\\(?:display|text)style\s*(.*)\}$", re.DOTALL)
# Past this depth lists stop indenting, so hostile markup cannot multiply the text it reads as
# (tables keep their own budgets, ``_tables``).
_MAX_INDENT = 10


@dataclass(frozen=True, slots=True, kw_only=True)
class Heading:
    """A heading of the converted text.

    Attributes:
        level: 1 to 6, as in ``h1``…``h6``.
        title: Its text.
        anchor: Its ``id``, the page's anchor for it.
        start: Where its line starts in the text.
    """

    level: int
    title: str
    anchor: str
    start: int


@dataclass(frozen=True, slots=True)
class WikiText:
    """A page's text and its headings, in order."""

    text: str
    headings: tuple[Heading, ...]


def wiki_text(html: str) -> WikiText:
    """The text of a MediaWiki page's rendered HTML, and its headings."""
    converter = _Converter()
    converter.feed(html)
    converter.close()
    return converter.result()


@dataclass(slots=True)
class _Table:
    rows: list[list[Cell]] = field(default_factory=list)
    caption: list[str] = field(default_factory=list)
    cell: Cell | None = None
    in_caption: bool = False

    def lines(self) -> list[str]:
        return table_lines(self.rows, " ".join(" ".join(self.caption).split()))


def _classes(attributes: dict[str, str | None]) -> list[str]:
    return (attributes.get("class") or "").split()


def _number(value: str | None) -> int | None:
    """An attribute's integer (``start``, ``value``); ``None`` when it is not one."""
    try:
        return int(value or "")
    except ValueError:
        return None


class _Converter(HTMLParser):
    """Walks the HTML once: blocks become lines, tables rows, headings ``#`` lines."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._lines: list[str] = []
        self._length = 0  # characters in ``_lines`` joined by newlines
        self._headings: list[Heading] = []
        self._inline: list[str] = []
        self._open: list[tuple[str, bool]] = []  # (tag, dropped)
        self._counts: Counter[str] = Counter()  # open elements by tag, for stray end tags
        self._dropped = 0
        self._lists: list[list[int | str]] = []  # [kind, items so far]
        self._dls = 0  # open definition lists, which indent their ``dd``
        self._item_prefix = ""
        self._tables: list[_Table] = []
        self._heading: tuple[int, str] | None = None  # (level, anchor) of the open heading
        self._pre = 0

    # --- the parser's events -----------------------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if self._dropped or self._is_noise(tag, attributes):
            if tag not in _VOID:
                self._push(tag, dropped=True)
            return
        if tag == "math":  # a formula reads as its TeX; its MathML would read as noise
            self._tex(attributes.get("alttext"))
            self._push(tag, dropped=True)
            return
        if tag in _VOID:
            if tag == "br":
                self._break()
            elif tag == "img" and _MATH_IMAGES.intersection(_classes(attributes)):
                self._tex(attributes.get("alt"))
            return
        self._push(tag, dropped=False)
        self._start(tag, attributes)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID or not self._counts[tag]:
            return  # a stray end tag
        while self._open:
            open_tag, dropped = self._open.pop()
            self._counts[open_tag] -= 1
            if dropped:
                self._dropped -= 1
            else:
                self._end(open_tag)
            if open_tag == tag:
                return

    def handle_data(self, data: str) -> None:
        if not self._dropped:
            self._inline.append(data)

    def close(self) -> None:
        super().close()
        while self._open:
            self.handle_endtag(self._open[-1][0])
        self._flush()

    def result(self) -> WikiText:
        return WikiText("\n".join(self._lines), tuple(self._headings))

    # --- structure -----------------------------------------------------------------------------

    def _push(self, tag: str, *, dropped: bool) -> None:
        self._open.append((tag, dropped))
        self._counts[tag] += 1
        self._dropped += dropped

    def _is_noise(self, tag: str, attributes: dict[str, str | None]) -> bool:
        if tag in _NOISE_TAGS or attributes.get("role") == "navigation":
            return True
        if attributes.get("id") == "toc":  # the table of contents, whatever its classes
            return True
        if _NOISE.intersection(_classes(attributes)):
            return True
        return bool(_HIDDEN_STYLE.search(attributes.get("style") or ""))

    def _start(self, tag: str, attributes: dict[str, str | None]) -> None:
        if tag == "span":
            self._anchor(attributes)
        elif tag in _HEADINGS:
            self._flush()
            if not self._tables:  # a heading in a table is a value of its cell
                self._heading = (_HEADINGS[tag], attributes.get("id") or "")
        elif tag in _TABLE_PARTS:
            self._start_table_part(tag, attributes)
        elif tag in _LIST_PARTS:
            self._start_list_part(tag, attributes)
        elif tag == "pre":
            self._flush()
            self._pre += 1
        elif tag in _BLOCKS:
            self._flush()
            self._dls += tag == "dl"

    def _end(self, tag: str) -> None:
        if tag in _HEADINGS:
            if self._heading is not None:
                self._close_heading()
            else:
                self._flush()
        elif tag in _TABLE_PARTS:
            self._end_table_part(tag)
        elif tag in _LIST_PARTS:
            self._flush()
            if tag in ("ul", "ol") and self._lists:
                self._lists.pop()
            self._item_prefix = self._continuation()
        elif tag == "pre":
            self._flush()
            self._pre = max(self._pre - 1, 0)
        elif tag in _BLOCKS:
            self._flush()
            if tag == "dl":
                self._dls = max(self._dls - 1, 0)
                self._item_prefix = self._continuation()

    def _anchor(self, attributes: dict[str, str | None]) -> None:
        """The older heading markup puts the anchor on an inner ``span.mw-headline``."""
        heading = self._heading
        if heading is not None and not heading[1] and "mw-headline" in _classes(attributes):
            self._heading = (heading[0], attributes.get("id") or "")

    def _close_heading(self) -> None:
        if self._heading is None:
            return
        level, anchor = self._heading
        title = self._collapsed()
        self._inline.clear()
        self._heading = None
        if title:
            self._headings.append(
                Heading(level=level, title=title, anchor=anchor, start=self._next_start())
            )
            self._emit(f"{'#' * level} {title}")

    def _start_table_part(self, tag: str, attributes: dict[str, str | None]) -> None:
        if tag == "table":
            self._flush()
            self._tables.append(_Table())
            return
        if not self._tables:  # a table part outside a table is the HTML's slip
            return
        table = self._tables[-1]
        if tag == "tr":
            table.rows.append([])
            return
        self._flush()
        if tag == "caption":
            table.in_caption = True
            return
        if not table.rows:
            table.rows.append([])
        table.cell = cell(attributes.get("rowspan"), attributes.get("colspan"))
        table.rows[-1].append(table.cell)

    def _end_table_part(self, tag: str) -> None:
        if not self._tables:
            return
        self._flush()
        if tag == "caption":
            self._tables[-1].in_caption = False
        elif tag in ("td", "th"):
            self._tables[-1].cell = None
        elif tag == "table":
            lines = self._tables.pop().lines()
            if self._tables and self._tables[-1].cell is not None:
                self._tables[-1].cell.parts.append("; ".join(lines))  # a table in a cell
            else:
                for line in lines:
                    self._emit(line)

    def _start_list_part(self, tag: str, attributes: dict[str, str | None]) -> None:
        self._flush()
        if tag in ("ul", "ol"):
            start = _number(attributes.get("start"))
            self._lists.append([tag, (1 if start is None else start) - 1])
        elif tag == "li":
            value = _number(attributes.get("value"))
            if value is not None and self._lists:  # ``<li value>`` renumbers from here
                self._lists[-1][1] = value - 1
            self._item_prefix = self._next_item()
        else:  # dd
            self._item_prefix = self._indent(self._depth())

    def _next_item(self) -> str:
        indent = self._indent(self._depth() - 1)
        if not self._lists:
            return f"{indent}- "
        kind, count = self._lists[-1]
        self._lists[-1][1] = int(count) + 1
        return f"{indent}{int(count) + 1}. " if kind == "ol" else f"{indent}- "

    def _depth(self) -> int:
        """How deep the open lists and definition lists nest."""
        return len(self._lists) + self._dls

    def _indent(self, depth: int) -> str:
        return "  " * min(max(depth, 0), _MAX_INDENT)

    def _continuation(self) -> str:
        """The indent of a further line of the open list item or definition."""
        return self._indent(self._depth())

    # --- text ----------------------------------------------------------------------------------

    def _tex(self, source: str | None) -> None:
        """A formula, as the TeX the wiki gives (``{\\displaystyle …}`` unwrapped)."""
        tex = (source or "").strip()
        if match := _TEX_STYLE.match(tex):
            tex = match[1].strip()
        if tex:
            self._inline.append(f" {tex} ")

    def _collapsed(self) -> str:
        return " ".join("".join(self._inline).split())

    def _break(self) -> None:
        """A ``<br>``: the end of a line, or of a value in a table cell."""
        self._flush()

    def _flush(self) -> None:
        """The inline text so far ends a line (or a value of the open cell or caption)."""
        text = "".join(self._inline).strip("\n") if self._pre else self._collapsed()
        self._inline.clear()
        if not text:
            return
        table = self._tables[-1] if self._tables else None
        if table is not None:  # a cell's or a caption's value keeps to its row
            text = " ".join(text.split())
        if table is not None and table.in_caption:
            table.caption.append(text)
        elif table is not None and table.cell is not None:
            table.cell.parts.append(text)
        elif table is not None:
            return  # text between a table's cells is the HTML's slip, not content
        else:
            self._emit(f"{self._item_prefix}{text}")
            self._item_prefix = self._continuation()

    def _next_start(self) -> int:
        return self._length + (1 if self._lines else 0)

    def _emit(self, line: str) -> None:
        self._length = self._next_start() + len(line)
        self._lines.append(line)
