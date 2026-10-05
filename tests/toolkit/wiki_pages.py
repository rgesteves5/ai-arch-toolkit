"""MediaWiki answers as the Action API sends them (``formatversion=2``), for the wiki family's
tests and contract cases.

The HTML follows the parser output the documentation shows: the ``mw-parser-output`` wrapper,
headings in ``div.mw-heading`` with the anchor on the ``h2`` (https://www.mediawiki.org/wiki/
Heading_HTML_changes), edit links in ``span.mw-editsection``, footnote markers in
``sup.reference`` and their list in ``ol.references``.
"""

from __future__ import annotations

from typing import Any

_DOCREF = "See https://en.wikipedia.org/w/api.php for API usage."


def api_error(code: str, info: str) -> dict[str, Any]:
    """An error answer, as MediaWiki sends it with HTTP 200."""
    return {
        "error": {"code": code, "info": info, "docref": _DOCREF},
        "servedby": "mw-api-ext.eqiad.main-79f4dd7c47-hgftf",
    }


MISSING_PAGE = api_error("missingtitle", "The page you specified doesn't exist.")


def heading(level: int, title: str, anchor: str = "") -> str:
    """A heading as the parser renders it today, with its edit link."""
    anchor = anchor or title.replace(" ", "_")
    return (
        f'<div class="mw-heading mw-heading{level}"><h{level} id="{anchor}">{title}</h{level}>'
        '<span class="mw-editsection"><span class="mw-editsection-bracket">[</span>'
        f'<a href="/w/index.php?title=X&amp;action=edit&amp;section=1" title="Edit section: '
        f'{title}"><span>edit</span></a><span class="mw-editsection-bracket">]</span></span></div>'
    )


def page_html(*parts: str) -> str:
    """A page's rendered HTML: its parts inside the parser's wrapper."""
    body = "\n".join(parts)
    return f'<div class="mw-content-ltr mw-parser-output" lang="en" dir="ltr">{body}</div>'


def parse_answer(title: str, html: str = "", *, redirected_from: str = "") -> dict[str, Any]:
    """An ``action=parse&prop=text`` answer."""
    parse: dict[str, Any] = {"title": title, "pageid": 4242, "text": html}
    if redirected_from:
        parse["redirects"] = [{"from": redirected_from, "to": title}]
    return {"parse": parse}


def search_answer(
    titles: list[str],
    *,
    total: int,
    next_offset: int | None = None,
    suggestion: str = "",
) -> dict[str, Any]:
    """A ``list=search`` answer."""
    info: dict[str, Any] = {"totalhits": total}
    if suggestion:
        info["suggestion"] = suggestion
    answer: dict[str, Any] = {
        "batchcomplete": True,
        "query": {
            "searchinfo": info,
            "search": [
                {
                    "ns": 0,
                    "title": title,
                    "pageid": 100 + number,
                    "size": 5000,
                    "wordcount": 800,
                    "snippet": f'the <span class="searchmatch">{title}</span> page',
                    "timestamp": "2026-09-01T10:00:00Z",
                }
                for number, title in enumerate(titles)
            ],
        },
    }
    if next_offset is not None:
        answer["continue"] = {"sroffset": next_offset, "continue": "-||"}
    return answer


def _reference(number: int) -> str:
    return (
        f'<sup id="cite_ref-{number}" class="reference"><a href="#cite_note-{number}">'
        f'<span class="cite-bracket">[</span>{number}<span class="cite-bracket">]</span>'
        "</a></sup>"
    )


# The page of the first conversation: the 1960 prize's rationale sits in a table row, far into
# the page (character 13 888 of the wikitext), in a table where shared years and rationales span
# rows.
NOBEL_TITLE = "List of Nobel laureates in Physics"
_LAUREATES = [
    ("1956", 3, "William Shockley", "United States", "for their researches on semiconductors", 3),
    ("", 0, "John Bardeen", "United States", "", 0),
    ("", 0, "Walter Houser Brattain", "United States", "", 0),
    (
        "1957",
        2,
        "Chen Ning Yang",
        "China",
        "for their penetrating investigation of the parity laws",
        2,
    ),
    ("", 0, "Tsung-Dao Lee", "China", "", 0),
    (
        "1958",
        3,
        "Pavel Cherenkov",
        "Soviet Union",
        "for the discovery and the interpretation of the Cherenkov effect",
        3,
    ),
    ("", 0, "Ilya Frank", "Soviet Union", "", 0),
    ("", 0, "Igor Tamm", "Soviet Union", "", 0),
    ("1959", 2, "Emilio Segrè", "Italy", "for their discovery of the antiproton", 2),
    ("", 0, "Owen Chamberlain", "United States", "", 0),
    ("1960", 1, "Donald A. Glaser", "United States", "for the invention of the bubble chamber", 1),
    (
        "1961",
        1,
        "Robert Hofstadter",
        "United States",
        "for his pioneering studies of electron scattering in atomic nuclei",
        1,
    ),
]


def _laureate_row(row: tuple[str, int, str, str, str, int]) -> str:
    year, year_span, name, country, rationale, rationale_span = row
    cells = []
    if year:
        cells.append(f'<td rowspan="{year_span}" style="text-align:center">{year}</td>')
    cells.append('<td><span typeof="mw:File"><img src="//upload.wikimedia.org/x.jpg" '
                 'width="80" height="100"></span></td>')  # fmt: skip
    cells.append(f'<th scope="row"><a href="/wiki/{name}">{name}</a></th>')
    cells.append(f"<td>{country}</td>")
    if rationale:
        cells.append(f'<td rowspan="{rationale_span}">"{rationale}"{_reference(7)}</td>')
    return "<tr>" + "".join(cells) + "</tr>"


NOBEL_HTML = page_html(
    '<div class="shortdescription nomobile noexcerpt noprint searchaux" style="display:none">'
    "None</div>",
    '<style data-mw-deduplicate="TemplateStyles:r1236090951">.mw-parser-output .hatnote'
    "{font-style:italic}</style>",
    '<div role="note" class="hatnote navigation-not-searchable">For the prize, see '
    '<a href="/wiki/Nobel_Prize_in_Physics">Nobel Prize in Physics</a>.</div>',
    "<p>The <b>Nobel Prize in Physics</b> is awarded annually by the Royal Swedish Academy of "
    f"Sciences.{_reference(1)}</p>",
    "<p>" + "History of the prize and of its laureates. " * 300 + "</p>",
    heading(2, "Laureates"),
    '<table class="wikitable sortable"><tbody>'
    "<tr><th>Year</th><th>Image</th><th>Laureate</th><th>Country</th><th>Rationale</th></tr>"
    + "".join(_laureate_row(row) for row in _LAUREATES)
    + '<tr><td style="text-align:center">1940</td><td colspan="4">Not awarded</td></tr>'
    "</tbody></table>",
    heading(2, "References"),
    '<div class="reflist"><div class="mw-references-wrap"><ol class="references">'
    '<li id="cite_note-1"><span class="mw-cite-backlink"><a href="#cite_ref-1">^</a></span> '
    '<span class="reference-text">Nobel Foundation.</span></li></ol></div></div>',
    '<div class="navbox-styles"><style>.navbox{border:1px}</style></div>'
    '<div role="navigation" class="navbox" aria-labelledby="Nobel"><table class="nowraplinks">'
    "<tr><th>Nobel Prize in Physics laureates</th></tr></table></div>",
)

# The page of the second conversation: a Wikibooks book whose parts are sections of one page,
# not subpages.
WIKIBOOK_TITLE = "Creative Writing/Novels"
WIKIBOOK_SECTIONS = ["Characters", "Style", "Editing", "Publishing"]
WIKIBOOK_HTML = page_html(
    "<p>A novel is a long work of narrative fiction.</p>",
    *(
        f"{heading(2, name)}<p>What a novelist should know about {name.lower()}. "
        + f"More on {name.lower()}. " * 20
        + "</p>"
        for name in WIKIBOOK_SECTIONS
    ),
)


def long_page(paragraphs: int = 60) -> str:
    """A page long enough to need several windows."""
    return page_html(
        *(f"<p>Paragraph {number}: {'words of the page ' * 8}</p>" for number in range(paragraphs))
    )


def many_sections(count: int = 60) -> str:
    """A page with more sections than one outline window lists."""
    names = [f"Part {number}" for number in range(1, count + 1)]
    return page_html(
        "<p>Intro.</p>", *(f"{heading(2, name)}<p>Text of {name}.</p>" for name in names)
    )


# A Wiktionary entry: languages as level-2 headings, parts of speech below, senses as an ordered
# list with examples nested (https://en.wiktionary.org/wiki/Wiktionary:Entry_layout).
ENTRY_TERM = "serendipity"
ENTRY_HTML = page_html(
    heading(2, "English"),
    heading(3, "Etymology"),
    "<p>Coined by Horace Walpole in 1754, after the Persian fairy tale <i>The Three Princes of "
    "Serendip</i>.</p>",
    heading(3, "Noun"),
    '<p><span class="headword-line"><strong class="Latn headword" lang="en">serendipity</strong>'
    " (<i>countable and uncountable</i>, <i>plural</i> serendipities)</span></p>",
    "<ol><li>A combination of events which have come together by chance to make a surprisingly "
    'good or wonderful outcome.<dl><dd><div class="h-usage-example"><i>It was pure serendipity '
    "that we met.</i></div></dd></dl></li>"
    "<li>An unsought, unintended, and unexpected, but fortunate, discovery.</li></ol>",
    heading(2, "French"),
    heading(3, "Noun", "Noun_2"),
    "<ol><li>serendipity</li></ol>",
)


def long_entry(paragraphs: int = 60) -> str:
    """An entry whose English section needs several windows."""
    return page_html(
        heading(2, "English"),
        *(f"<p>Sense {number}: {'words of the entry ' * 8}</p>" for number in range(paragraphs)),
        heading(2, "French"),
        "<p>Le mot.</p>",
    )
