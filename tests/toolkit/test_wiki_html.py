"""Tests for toolkit/tools/_wiki_html.py: MediaWiki's rendered HTML as text a model reads."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.toolkit.tools._wiki_html import Heading, wiki_text
from tests.toolkit.wiki_pages import NOBEL_HTML, heading, page_html


def _text(*parts: str) -> str:
    return wiki_text(page_html(*parts)).text


class TestHeadings:
    def test_todays_markup_gives_the_level_the_title_and_the_anchor(self) -> None:
        converted = wiki_text(page_html("<p>Intro.</p>", heading(2, "Early life", "Early_life")))

        assert converted.text == "Intro.\n## Early life"
        assert converted.headings == (
            Heading(level=2, title="Early life", anchor="Early_life", start=7),
        )

    def test_the_older_markup_keeps_the_anchor_on_its_inner_span(self) -> None:
        older = (
            '<h3><span class="mw-headline" id="Later_years">Later years</span>'
            '<span class="mw-editsection">[<a href="x">edit</a>]</span></h3>'
        )

        converted = wiki_text(page_html(older))

        assert converted.text == "### Later years"
        assert converted.headings == (
            Heading(level=3, title="Later years", anchor="Later_years", start=0),
        )

    def test_parsoid_markup_reads_the_same(self) -> None:
        # Parsoid wraps each section in <section> and footnotes in sup.mw-ref.reference
        # (https://www.mediawiki.org/wiki/Specs/HTML; seen on en.wikipedia.org's read views).
        parsoid = (
            '<div class="mw-content-ltr mw-parser-output" data-mw-parsoid-version="0.24.0">'
            '<section data-mw-section-id="0" id="mwAQ"><p>Lead<sup class="mw-ref reference" '
            'typeof="mw:Extension/ref"><a href="#cite_note-1">[1]</a></sup>.</p></section>'
            '<section data-mw-section-id="1" id="mwGQ" aria-labelledby="Rules">'
            '<div class="mw-heading mw-heading2 ext-discussiontools-init-section">'
            '<h2 id="Rules"><span id="Old_rules" typeof="mw:FallbackId"></span>Rules</h2>'
            '<span class="mw-editsection">[edit]</span></div><p>One rule.</p>'
            '<link rel="mw:PageProp/Category" href="./Category:X"></section></div>'
        )

        converted = wiki_text(parsoid)

        assert converted.text == "Lead.\n## Rules\nOne rule."
        assert converted.headings == (Heading(level=2, title="Rules", anchor="Rules", start=6),)

    def test_the_table_of_contents_and_its_heading_are_left_out(self) -> None:
        toc = (
            '<div id="toc" role="navigation" aria-labelledby="mw-toc-heading">'
            '<h2 id="mw-toc-heading">Contents</h2><ul><li>1 History</li></ul></div>'
        )

        converted = wiki_text(page_html(toc, heading(2, "History")))

        assert converted.text == "## History"
        assert [found.title for found in converted.headings] == ["History"]

    def test_each_heading_knows_where_its_line_starts(self) -> None:
        converted = wiki_text(
            page_html("<p>A.</p>", heading(2, "One"), "<p>B.</p>", heading(3, "Two"))
        )

        for found in converted.headings:
            assert converted.text[found.start :].startswith("#" * found.level + " " + found.title)


class TestWhatIsLeftOut:
    @pytest.mark.parametrize(
        "noise",
        [
            '<sup class="reference"><a href="#n1">[1]</a></sup>',
            '<span class="mw-editsection">[edit]</span>',
            '<div id="toc" class="toc"><h2>Contents</h2><ul><li>1 History</li></ul></div>',
            '<div class="navbox" role="navigation"><table><tr><td>Links</td></tr></table></div>',
            '<div role="navigation" class="sidebar">Series</div>',
            '<table class="ambox"><tr><td>This article needs citations.</td></tr></table>',
            '<div class="shortdescription" style="display:none">Short</div>',
            '<span style="display: none">sort key</span>',
            '<span class="noprint">[citation needed]</span>',
            '<p class="mw-empty-elt"></p>',
            "<style>.mw-parser-output .x{color:red}</style>",
            "<script>alert(1)</script>",
            '<link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r1">',
            '<ol class="references"><li><span class="reference-text">A source.</span></li></ol>',
        ],
    )
    def test_navigation_maintenance_and_hidden_parts(self, noise: str) -> None:
        assert _text(f"<p>Kept{noise} text.</p>") == "Kept text."

    def test_a_hatnote_stays_it_names_the_page_to_read_next(self) -> None:
        note = '<div role="note" class="hatnote">Main article: History of physics</div>'

        assert _text(note) == "Main article: History of physics"


class TestTables:
    def test_a_cell_that_spans_rows_repeats_so_every_row_reads_whole(self) -> None:
        table = (
            '<table class="wikitable"><tr><th>Year</th><th>Laureate</th><th>Why</th></tr>'
            '<tr><td rowspan="2">1959</td><td>Segrè</td><td rowspan="2">antiproton</td></tr>'
            "<tr><td>Chamberlain</td></tr></table>"
        )

        assert _text(table).splitlines() == [
            "Year | Laureate | Why",
            "1959 | Segrè | antiproton",
            "1959 | Chamberlain | antiproton",
        ]

    def test_a_cell_that_spans_columns_leaves_the_others_empty(self) -> None:
        table = (
            '<table><tr><td>1940</td><td colspan="3">Not awarded</td></tr>'
            "<tr><td>1941</td><td>a</td><td>b</td><td>c</td></tr></table>"
        )

        assert _text(table).splitlines() == ["1940 | Not awarded", "1941 | a | b | c"]

    def test_the_caption_and_a_table_inside_a_cell(self) -> None:
        table = (
            "<table><caption>Results</caption><tr><td>Outer</td><td>"
            "<table><tr><td>in 1</td><td>in 2</td></tr></table></td></tr></table>"
        )

        assert _text(table).splitlines() == ["Table: Results", "Outer | in 1 | in 2"]

    def test_line_breaks_in_a_cell_separate_its_values(self) -> None:
        table = "<table><tr><td>Born</td><td>1879<br>Ulm, Germany</td></tr></table>"

        assert _text(table) == "Born | 1879; Ulm, Germany"

    @pytest.mark.parametrize("span", ["1000000", "-3", "two", ""])
    def test_a_span_the_html_gets_wrong_stays_bounded(self, span: str) -> None:
        table = (
            f'<table><tr><td rowspan="{span}" colspan="{span}">x</td><td>y</td></tr>'
            "<tr><td>z</td></tr></table>"
        )

        lines = _text(table).splitlines()

        assert len(lines) == 2
        assert all(line.count("|") < 60 for line in lines)

    def test_spans_stop_repeating_past_the_tables_budget(self) -> None:
        # 9 KB of spans would lay out 2.5 million cells (review of T05).
        first = "<tr>" + '<td colspan="50" rowspan="500">x</td>' * 100 + "</tr>"
        table = "<table>" + first + "<tr><td>y</td></tr>" * 499 + "</table>"

        text = _text(table)

        assert len(text) < 1_000_000
        assert text.count("\n") == 499  # every row still reads

    def test_a_heading_or_preformatted_text_in_a_cell_stays_in_its_row(self) -> None:
        table = (
            "<table><tr><td><h3>Title</h3>text</td><td><pre>a = 1\nb = 2</pre></td>"
            "<td>c</td></tr></table>"
        )

        assert _text(table) == "Title; text | a = 1 b = 2 | c"

    def test_the_nobel_row_of_1960_reads_with_its_rationale(self) -> None:
        text = wiki_text(NOBEL_HTML).text

        row = next(line for line in text.splitlines() if line.startswith("1960 |"))
        assert row == (
            "1960 |  | Donald A. Glaser | United States | "
            '"for the invention of the bubble chamber"'
        )
        assert "1956 |  | Walter Houser Brattain | United States | " in text
        assert "Nobel Prize in Physics laureates" not in text  # the navbox
        assert "Nobel Foundation." not in text  # the references


class TestText:
    def test_lists_keep_their_depth_numbers_and_further_lines(self) -> None:
        lists = (
            "<ul><li>One<ul><li>Nested</li></ul>more of one</li><li>Two</li></ul>"
            '<ol start="3"><li>Three</li><li>Four</li></ol>'
            "<dl><dt>Term</dt><dd>Meaning</dd></dl>"
        )

        assert _text(lists).splitlines() == [
            "- One",
            "  - Nested",
            "  more of one",
            "- Two",
            "3. Three",
            "4. Four",
            "Term",
            "  Meaning",
        ]

    def test_numbering_follows_start_and_value(self) -> None:
        lists = '<ol start="0"><li>zero</li><li value="10">ten</li><li>eleven</li></ol>'

        assert _text(lists).splitlines() == ["0. zero", "10. ten", "11. eleven"]

    def test_nested_definitions_keep_their_depth(self) -> None:
        thread = "<dl><dd>first<dl><dd>reply<dl><dd>answer</dd></dl></dd></dl></dd></dl>"

        assert _text(thread).splitlines() == ["  first", "    reply", "      answer"]

    def test_deep_nesting_stops_indenting(self) -> None:
        # Indenting every level would grow the text with the square of the depth.
        deep = "<ul><li>x" * 2000 + "</li></ul>" * 2000

        lines = _text(deep).splitlines()

        assert len(lines) == 2000
        assert max(len(line) - len(line.lstrip()) for line in lines) == 2 * 10

    def test_stray_end_tags_are_ignored(self) -> None:
        assert _text("<p>a</p>" + "</div></span>" * 20_000 + "<p>b</p>") == "a\nb"

    def test_inline_markup_reads_as_its_words(self) -> None:
        paragraph = (
            '<p>The <b>bubble</b> <a href="/wiki/C">chamber</a>&nbsp;was in<i>vented</i>.</p>'
        )

        assert _text(paragraph) == "The bubble chamber was invented."

    @pytest.mark.parametrize(
        "formula",
        [
            '<math alttext="{\\displaystyle E=mc^{2}}"><mi>E</mi></math>',
            '<span class="mwe-math-element"><span class="mwe-math-mathml-inline" '
            'style="display: none;"><math alttext="x"><mi>x</mi></math></span>'
            '<img class="mwe-math-fallback-image-inline" alt="{\\displaystyle E=mc^{2}}"></span>',
        ],
    )
    def test_a_formula_reads_as_its_tex(self, formula: str) -> None:
        assert _text(f"<p>So {formula} holds.</p>") == "So E=mc^{2} holds."

    def test_preformatted_text_keeps_its_lines(self) -> None:
        assert _text("<pre>a = 1\n  b = 2</pre>") == "a = 1\n  b = 2"

    def test_a_figure_reads_as_its_caption(self) -> None:
        figure = (
            '<figure typeof="mw:File/Thumb"><a href="x"><img src="y.jpg"></a>'
            "<figcaption>The first bubble chamber.</figcaption></figure>"
        )

        assert _text(figure) == "The first bubble chamber."

    @pytest.mark.parametrize(
        "broken",
        [
            "<p>Unclosed <b>bold",
            "</div></td></table><p>Stray ends</p>",
            "<table><tr><td>cell<p>para</td></tr>",
            "<ul><li>item<li>next</ul>",
        ],
    )
    def test_broken_html_still_gives_text(self, broken: str) -> None:
        assert wiki_text(broken).text.strip()
