"""A DailyMed label, an SPL document, as text: its facts and its numbered sections (T07).

An SPL is an HL7 v3 document (namespace ``urn:hl7-org:v3``) whose body holds sections, each with a
LOINC ``code`` and its ``displayName``, a ``title``, a narrative ``text`` and its subsections under
``component/section`` (https://www.fda.gov/industry/structured-product-labeling/section-headings-loinc).
The narrative is CDA's: ``paragraph``, ``list`` of ``item``, ``table`` with a ``caption`` and rows
of ``th``/``td``, and inline ``content``, ``sub``, ``sup``, ``linkHtml``, ``footnote`` and ``br``;
an image (``renderMultiMedia``) has no text (https://hl7.org/cda/stds/core/narrative.html).

The label's text has every section in document order under a heading (``##`` for a top-level
section, one ``#`` more per level), lists one item per line, and tables one row per line with
`` | `` between cells. A section is numbered by its place in that order, and its text runs to the
end of its last subsection: what ``dailymed_label_text(section=N)`` returns.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass

_V3 = "{urn:hl7-org:v3}"
_NS = {"v3": "urn:hl7-org:v3"}
_MAX_INDENT = 10
# Narrative blocks, each on lines of its own; cells and items, set apart by spaces when inlined.
_BLOCKS = frozenset({"paragraph", "list", "table"})
_CELLS = frozenset({"item", "td", "th", "tr", "caption"})
_ROW_GROUPS = frozenset({"thead", "tbody", "tfoot"})
# A ``br``: XML cannot hold NUL, so it never comes from the text.
_BREAK = "\x00"


@dataclass(frozen=True, slots=True, kw_only=True)
class SplSection:
    """A section of a label.

    Attributes:
        number: Its place in the label, from 1, subsections included.
        depth: 0 for a top-level section, one more per level below.
        title: Its title, or its LOINC name when it has none.
        code: Its LOINC code and name, e.g. ``"34067-9: INDICATIONS & USAGE SECTION"``.
        start: Where its text starts in the label's text (at its heading).
        end: Where its text ends, after its last subsection.
    """

    number: int
    depth: int
    title: str
    code: str
    start: int
    end: int


@dataclass(frozen=True, slots=True, kw_only=True)
class SplLabel:
    """A label as the tools read it.

    Attributes:
        title: The document's title.
        effective: The version's effective date (ISO 8601), or the source's value.
        version: The version number.
        organization: The labeler.
        text: Every section's text, in document order.
        sections: The sections, numbered in that order.
    """

    title: str
    effective: str
    version: str
    organization: str
    text: str
    sections: tuple[SplSection, ...]


def spl_label(xml_text: str) -> SplLabel:
    """Read an SPL document.

    Raises:
        xml.etree.ElementTree.ParseError: The text is not XML (a ``SyntaxError``, which the HTTP
            door turns into the API's failure).
    """
    root = ET.fromstring(xml_text)
    writer = _Writer()
    body = root.find("v3:component/v3:structuredBody", _NS)
    for element in _subsections(body):
        writer.section(element, 0)
    organization = root.find(".//v3:representedOrganization/v3:name", _NS)
    return SplLabel(
        title=_inline(root.find("v3:title", _NS)),
        effective=_date(_attribute(root.find("v3:effectiveTime", _NS), "value")),
        version=_attribute(root.find("v3:versionNumber", _NS), "value"),
        organization=_inline(organization),
        text=writer.text(),
        sections=tuple(writer.sections),
    )


class _Writer:
    """The label's text as it is written, and where each section starts and ends."""

    __slots__ = ("lines", "sections", "size")

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.size = 0
        self.sections: list[SplSection] = []

    def add(self, line: str) -> None:
        self.lines.append(line)
        self.size += len(line) + 1

    def text(self) -> str:
        return "".join(f"{line}\n" for line in self.lines)

    def section(self, element: ET.Element, depth: int) -> None:
        """Write ``element``'s heading, narrative and subsections, numbered in order."""
        code = element.find("v3:code", _NS)
        name = _attribute(code, "displayName")
        title = _inline(element.find("v3:title", _NS)) or name or "(untitled)"
        index, start = len(self.sections), self.size
        self.sections.append(SplSection(number=0, depth=0, title="", code="", start=0, end=0))
        self.add(f"{'#' * min(depth + 2, 6)} {title}")
        for line in _blocks(element.find("v3:text", _NS), 0):
            self.add(line)
        for subsection in _subsections(element):
            self.section(subsection, depth + 1)
        loinc = _attribute(code, "code")
        self.sections[index] = SplSection(
            number=index + 1,
            depth=depth,
            title=title,
            code=": ".join(part for part in (loinc, name) if part),
            start=start,
            end=self.size,
        )


def _subsections(element: ET.Element | None) -> Iterator[ET.Element]:
    if element is not None:
        yield from element.iterfind("v3:component/v3:section", _NS)


def _blocks(element: ET.Element | None, depth: int) -> list[str]:
    """The lines of a narrative ``text``: its paragraphs, lists and tables, and the inline text
    between them."""
    if element is None:
        return []
    lines: list[str] = []
    run = [element.text or ""]
    for child in element:
        tag = _tag(child)
        if tag in _BLOCKS:
            lines += _lines("".join(run))
            run = []
            if tag == "list":
                lines += _list(child, depth)
            elif tag == "table":
                lines += _table(child)
            else:
                lines += _lines("".join(_pieces(child, "")))
        else:
            run += _pieces(child, "")
        run.append(child.tail or "")
    return lines + _lines("".join(run))


def _list(element: ET.Element, depth: int) -> list[str]:
    ordered = element.get("listType") == "ordered"
    indent = "  " * min(depth, _MAX_INDENT)
    lines: list[str] = []
    items = (child for child in element if _tag(child) == "item")
    for number, item in enumerate(items, start=1):
        marker = f"{number}." if ordered else "-"
        lines.append(f"{indent}{marker} {_inline(item, skip='list')}".rstrip())
        for nested in item:
            if _tag(nested) == "list":
                lines += _list(nested, depth + 1)
    return lines


def _table(element: ET.Element) -> list[str]:
    """A table's caption and its own rows, one per line (a table in a cell stays in the cell, so
    each element is read once)."""
    caption = _inline(element.find("v3:caption", _NS))
    groups = [element, *(child for child in element if _tag(child) in _ROW_GROUPS)]
    rows = [
        " | ".join(_inline(cell) for cell in row if _tag(cell) in ("th", "td"))
        for group in groups
        for row in group
        if _tag(row) == "tr"
    ]
    return ([f"Table: {caption}"] if caption else []) + [row for row in rows if row.strip(" |")]


def _inline(element: ET.Element | None, *, skip: str = "") -> str:
    """An element's text with its descendants', whitespace collapsed, on one line: a ``br``
    becomes ``" / "``; ``skip`` names a child tag left out (an item's nested list)."""
    if element is None:
        return ""
    return " / ".join(_lines("".join(_pieces(element, skip))))


def _pieces(element: ET.Element, skip: str) -> Iterator[str]:
    """The text of ``element`` and its descendants in order (not its own tail): a ``br`` is a
    break, a block or a cell inside is set apart by spaces, an image has no text."""
    tag = _tag(element)
    if tag == "br":
        yield _BREAK
        return
    if tag == "renderMultiMedia":
        return
    space = " " if tag in _BLOCKS or tag in _CELLS else ""
    yield space + (element.text or "")
    for child in element:
        if _tag(child) != skip:
            yield from _pieces(child, "")
        yield child.tail or ""
    yield space


def _lines(text: str) -> list[str]:
    """The non-blank lines of narrative text, whitespace collapsed: breaks (``br``) end lines,
    the source's own newlines are whitespace."""
    return [line for line in (" ".join(raw.split()) for raw in text.split(_BREAK)) if line]


def _tag(element: ET.Element) -> str:
    return str(element.tag).removeprefix(_V3)


def _attribute(element: ET.Element | None, name: str) -> str:
    if element is None:
        return ""
    return " ".join(str(element.get(name, "")).split())


def _date(value: str) -> str:
    """An HL7 timestamp's date (``20260608`` or ``20260608120000-0500``) in ISO 8601."""
    digits = value[:8]
    if len(digits) == 8 and digits.isdigit():
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
    return value
