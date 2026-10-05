"""Tests for toolkit/tools/_wiki.py: the wiki family (T05).

The two conversations that opened the tools contract front are tests here: the rationale of the
1960 Nobel Prize in Physics, found in one ``find`` call, and a Wikibooks book read section by
section.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolFailure, ToolGroup, ToolResult
from ai_arch_toolkit.toolkit.tools import wiki_outline, wiki_read, wiki_search, wiktionary_entry
from tests.toolkit.http_fakes import HTTP_OPEN, respond
from tests.toolkit.wiki_pages import (
    ENTRY_HTML,
    ENTRY_TERM,
    MISSING_PAGE,
    NOBEL_HTML,
    NOBEL_TITLE,
    WIKIBOOK_HTML,
    WIKIBOOK_TITLE,
    api_error,
    heading,
    long_page,
    page_html,
    parse_answer,
    search_answer,
)


def _failure(fn: Any, *args: Any, **kwargs: Any) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


def _sent(mock_urlopen: MagicMock, call: int = -1) -> tuple[str, dict[str, list[str]]]:
    """The host and the query of a request the tool sent."""
    url = urlparse(mock_urlopen.call_args_list[call].args[0].full_url)
    return url.netloc, parse_qs(url.query)


class TestTheConversations:
    @patch(HTTP_OPEN)
    def test_the_1960_rationale_comes_in_one_find_call(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond(parse_answer(NOBEL_TITLE, NOBEL_HTML))

        result = wiki_read(NOBEL_TITLE, find="1960")

        text = _text(result)
        assert text.startswith(
            f"{NOBEL_TITLE} (en.wikipedia.org), passages that mention '1960':\n[at char "
        )
        assert (
            "1960 |  | Donald A. Glaser | United States | "
            '"for the invention of the bubble chamber"'
        ) in text
        assert text.endswith('[matches 1-1 of 1 for "1960" | end]')
        host, query = _sent(mock_urlopen)
        assert host == "en.wikipedia.org"
        assert query["action"] == ["parse"]
        assert query["page"] == [NOBEL_TITLE]
        assert query["prop"] == ["text"]
        assert query["formatversion"] == ["2"]
        assert query["redirects"] == ["1"]
        assert query["disabletoc"] == ["1"]
        assert "section" not in query  # a section is cut from the page, never asked for

    @patch(HTTP_OPEN)
    def test_a_wikibooks_book_reads_section_by_section(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = [
            respond(parse_answer(WIKIBOOK_TITLE, WIKIBOOK_HTML)) for _ in range(2)
        ]

        outline = _text(wiki_outline(WIKIBOOK_TITLE, wiki="en.wikibooks.org"))

        lines = outline.splitlines()
        assert lines[0] == (
            f"{WIKIBOOK_TITLE} (en.wikibooks.org), sections (read one with "
            "wiki_read(section=N, wiki='en.wikibooks.org')):"
        )
        assert lines[1].startswith("0. (introduction): ")
        assert [line.split(":")[0] for line in lines[2:]] == [
            "1. Characters",
            "2. Style",
            "3. Editing",
            "4. Publishing",
        ]
        host, query = _sent(mock_urlopen)
        assert host == "en.wikibooks.org"
        assert query["prop"] == ["text"]

        text = _text(wiki_read(WIKIBOOK_TITLE, wiki="en.wikibooks.org", section=3))

        assert text.splitlines()[:3] == [
            f"{WIKIBOOK_TITLE} (en.wikibooks.org), section 3 (Editing):",
            "## Editing",
            "What a novelist should know about editing. "
            + "More on editing. " * 19
            + "More on editing.",
        ]
        assert "Publishing" not in text

    @patch(HTTP_OPEN)
    def test_a_section_size_counts_its_subsections_as_wiki_read_returns_them(
        self, mock_urlopen: MagicMock
    ) -> None:
        html = page_html(
            "<p>Intro.</p>",
            heading(2, "A"),
            "<p>" + "a" * 100 + "</p>",
            heading(3, "A1"),
            "<p>" + "b" * 50 + "</p>",
            heading(2, "B"),
            "<p>c</p>",
        )
        mock_urlopen.side_effect = [respond(parse_answer("Page", html)) for _ in range(2)]

        lines = _text(wiki_outline("Page")).splitlines()[1:]
        section = wiki_read("Page", section=1, max_chars=20_000)

        a_size = len("## A") + 1 + 100 + 1 + len("### A1") + 1 + 50 + 1
        assert lines == [
            "0. (introduction): 7 chars",
            f"1. A: {a_size} chars",
            f"  2. A1: {len('### A1') + 1 + 50 + 1} chars",
            f"3. B: {len('## B') + 1 + 1} chars",
        ]
        assert isinstance(section, ToolResult)
        assert section.metadata["window"]["total"] == a_size


class TestReading:
    @patch(HTTP_OPEN)
    def test_a_long_page_reads_window_by_window(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = [respond(parse_answer("Long", long_page())) for _ in range(2)]

        first = wiki_read("Long", max_chars=1000)
        second = wiki_read("Long", max_chars=1000, offset=first.metadata["window"]["last"])

        assert isinstance(first, ToolResult)
        assert first.metadata["window"]["next_call"] == {
            "offset": first.metadata["window"]["last"]
        }
        assert first.value.splitlines()[-1].startswith("[chars 0-")
        assert second.metadata["window"]["first"] == first.metadata["window"]["last"]
        assert "Paragraph 0:" not in second.value

    @patch(HTTP_OPEN)
    def test_reading_on_in_a_section_names_the_section(self, mock_urlopen: MagicMock) -> None:
        long_section = page_html(
            heading(2, "Long"), *(f"<p>Line {n} of the section.</p>" for n in range(80))
        )
        mock_urlopen.return_value = respond(parse_answer("Page", long_section))

        result = wiki_read("Page", section=1, max_chars=500)

        assert isinstance(result, ToolResult)
        next_call = result.metadata["window"]["next_call"]
        assert next_call == {"offset": result.metadata["window"]["last"], "section": 1}
        assert result.value.endswith(f"next: offset={next_call['offset']}, section=1]")

    @pytest.mark.parametrize(
        ("max_chars", "kept"),
        [("1000", True), (499, False), (20_001, False)],
    )
    @patch(HTTP_OPEN)
    def test_max_chars_is_an_integer_within_its_limits_through_the_executor(
        self, mock_urlopen: MagicMock, max_chars: object, kept: bool
    ) -> None:
        # A PEP 695 alias once hid the Range from the schema: max_chars became a string.
        mock_urlopen.return_value = respond(parse_answer("Long", long_page()))
        call = ToolCall(id="c1", name="wiki_read", input={"title": "Long", "max_chars": max_chars})

        result = ToolGroup(wiki_read).execute(call)

        assert result.ok is kept
        if not kept:
            assert result.error is not None
            assert result.error.type == "validation_error"
            assert "from 500 to 20000" in result.error.message

    @patch(HTTP_OPEN)
    def test_a_term_the_page_does_not_have_says_so(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond(parse_answer(NOBEL_TITLE, NOBEL_HTML))

        text = _text(wiki_read(NOBEL_TITLE, find="2077"))

        assert text.endswith('[no matches for "2077"]')

    @patch(HTTP_OPEN)
    def test_a_redirect_is_named_in_the_heading(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond(
            parse_answer(
                "Python (programming language)",
                "<p>A language.</p>",
                redirected_from="Python lang",
            )
        )

        text = _text(wiki_read("Python lang"))

        assert text.splitlines()[0] == (
            "Python (programming language) (en.wikipedia.org, redirected from 'Python lang'):"
        )

    @patch(HTTP_OPEN)
    def test_an_introduction_without_text_says_so(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond(
            parse_answer("Page", page_html(heading(2, "First"), "<p>Text.</p>"))
        )

        assert _text(wiki_read("Page", section=0)) == (
            "Page (en.wikipedia.org), section 0 (introduction) has no text."
        )


class TestFailures:
    @pytest.mark.parametrize(
        ("call", "message"),
        [
            (
                lambda: wiki_read("No such page"),
                "en.wikipedia.org has no page titled 'No such page'; find the exact title with "
                "wiki_search",
            ),
            (
                lambda: wiki_outline("Creative Writing/Novels/Editing", wiki="en.wikibooks.org"),
                "en.wikibooks.org has no page titled 'Creative Writing/Novels/Editing'; find the "
                "exact title with wiki_search(wiki='en.wikibooks.org')",
            ),
            (
                lambda: wiktionary_entry("zzqqxx"),
                "en.wiktionary.org has no page titled 'zzqqxx'; find the exact title with "
                "wiki_search(wiki='en.wiktionary.org')",
            ),
        ],
    )
    @patch(HTTP_OPEN)
    def test_a_page_that_does_not_exist_is_not_found_with_the_next_step(
        self, mock_urlopen: MagicMock, call: Any, message: str
    ) -> None:
        mock_urlopen.return_value = respond(MISSING_PAGE)

        failure = _failure(call)

        assert failure.error.type == "not_found"
        assert failure.error.message == message

    @patch(HTTP_OPEN)
    def test_a_title_the_wiki_refuses_is_a_validation_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond(api_error("invalidtitle", 'Bad title "Talk:".'))

        failure = _failure(wiki_read, "Talk:")

        assert failure.error.type == "validation_error"
        assert 'Bad title "Talk:"' in failure.error.message

    @patch(HTTP_OPEN)
    def test_a_section_the_page_does_not_have_is_a_validation_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = respond(
            parse_answer("Page", page_html(heading(2, "One"), heading(2, "Two")))
        )

        failure = _failure(wiki_read, "Page", section=3)

        assert failure.error.type == "validation_error"
        assert failure.error.message == "Page has 2 sections, not 3; list them with wiki_outline"

    @patch(HTTP_OPEN)
    def test_a_special_page_is_a_validation_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond(
            api_error("pagecannotexist", "Namespace doesn't allow actual pages.")
        )

        failure = _failure(wiki_read, "Special:Random")

        assert failure.error.type == "validation_error"
        assert "not a page the wiki can hold" in failure.error.message

    @pytest.mark.parametrize("title", ["", "   ", "x" * 256, "é" * 128])
    @patch(HTTP_OPEN)
    def test_an_empty_or_too_long_title_fails_before_asking(
        self, mock_urlopen: MagicMock, title: str
    ) -> None:
        failure = _failure(wiki_read, title)

        assert failure.error.type == "validation_error"
        mock_urlopen.assert_not_called()

    @pytest.mark.parametrize(
        "call",
        [
            lambda: wiki_search("x", wiki="evil.example"),
            lambda: wiki_read("x", wiki="https://169.254.169.254/w/api.php"),
            lambda: wiki_outline("x", wiki="en.wikipedia.org.evil.example"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_a_wiki_off_wikimedia_is_refused_before_asking(
        self, mock_urlopen: MagicMock, call: Any
    ) -> None:
        failure = _failure(call)

        assert failure.error.type == "validation_error"
        assert "invalid wiki" in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_a_wiki_that_asks_to_slow_down_is_rate_limited(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond(
            api_error("ratelimited", "You've exceeded your rate limit.")
        )

        failure = _failure(wiki_search, "physics")

        assert failure.error.type == "rate_limited"
        assert failure.error.retryable


class TestSearch:
    @patch(HTTP_OPEN)
    def test_results_come_numbered_with_the_total_and_the_next_offset(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = respond(
            search_answer(["Bubble chamber", "Donald A. Glaser"], total=57, next_offset=12)
        )

        result = wiki_search("bubble chamber", max_results=2, offset=10)

        assert isinstance(result, ToolResult)
        assert result.value.splitlines() == [
            "Pages on en.wikipedia.org that match 'bubble chamber':",
            "11. Bubble chamber (800 words): the Bubble chamber page",
            "12. Donald A. Glaser (800 words): the Donald A. Glaser page",
            "[results 11-12 of 57 | next: offset=12]",
        ]
        _, query = _sent(mock_urlopen)
        assert query["list"] == ["search"]
        assert query["srsearch"] == ["bubble chamber"]
        assert query["srlimit"] == ["2"]
        assert query["sroffset"] == ["10"]
        assert query["srinfo"] == ["totalhits|suggestion"]

    @patch(HTTP_OPEN)
    def test_zero_results_say_so_with_the_query_and_the_wikis_suggestion(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = respond(
            search_answer([], total=0, suggestion="bubble chamber")
        )

        assert _text(wiki_search("bubbel chamber")) == (
            "No pages on en.wikipedia.org match 'bubbel chamber'. Did you mean 'bubble chamber'?"
        )

    @patch(HTTP_OPEN)
    def test_the_search_stops_at_the_depth_the_wiki_serves(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond(search_answer(["Last"], total=20_000))

        wiki_search("physics", max_results=10, offset=9_995)

        assert _sent(mock_urlopen)[1]["srlimit"] == ["5"]

    @patch(HTTP_OPEN)
    def test_an_empty_query_fails_before_asking(self, mock_urlopen: MagicMock) -> None:
        assert _failure(wiki_search, "  ").error.type == "validation_error"
        mock_urlopen.assert_not_called()


class TestWiktionary:
    @patch(HTTP_OPEN)
    def test_an_entry_reads_its_languages_section(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond(parse_answer(ENTRY_TERM, ENTRY_HTML))

        text = _text(wiktionary_entry(ENTRY_TERM))

        assert text.splitlines() == [
            "Wiktionary, serendipity (English):",
            "## English",
            "### Etymology",
            "Coined by Horace Walpole in 1754, after the Persian fairy tale The Three Princes of "
            "Serendip.",
            "### Noun",
            "serendipity (countable and uncountable, plural serendipities)",
            "1. A combination of events which have come together by chance to make a surprisingly "
            "good or wonderful outcome.",
            "    It was pure serendipity that we met.",
            "2. An unsought, unintended, and unexpected, but fortunate, discovery.",
        ]
        host, query = _sent(mock_urlopen)
        assert host == "en.wiktionary.org"
        assert (query["page"], query["prop"]) == ([ENTRY_TERM], ["text"])
        assert mock_urlopen.call_count == 1

    @patch(HTTP_OPEN)
    def test_a_language_the_entry_does_not_have_names_those_it_has(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = respond(parse_answer(ENTRY_TERM, ENTRY_HTML))

        failure = _failure(wiktionary_entry, ENTRY_TERM, language="Portuguese")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "Wiktionary's entry 'serendipity' has no Portuguese section; it has English, French"
        )

    @pytest.mark.parametrize("language", ["", "<b>", "x" * 81, "French|German"])
    @patch(HTTP_OPEN)
    def test_an_invalid_language_fails_before_asking(
        self, mock_urlopen: MagicMock, language: str
    ) -> None:
        failure = _failure(wiktionary_entry, ENTRY_TERM, language=language)

        assert failure.error.type == "validation_error"
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_a_language_name_with_letters_beyond_ascii(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = respond(
            parse_answer("hus", page_html(heading(2, "Norwegian Bokmål"), "<p>house</p>"))
        )

        assert "house" in _text(wiktionary_entry("hus", language="norwegian bokmål"))
