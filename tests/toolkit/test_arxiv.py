"""Tests for toolkit/tools/_arxiv.py (T06)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools import _arxiv
from ai_arch_toolkit.toolkit.tools._arxiv import arxiv_paper, arxiv_search
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond
from tests.toolkit.literature_answers import ARXIV_ERROR_FEED
from tests.toolkit.literature_answers import arxiv_entry as entry
from tests.toolkit.literature_answers import arxiv_feed as feed

_EMPTY = feed(total=0)


def _text(result: ToolResult) -> str:
    assert isinstance(result, ToolResult) and result.ok, result
    assert isinstance(result.value, str)
    return result.value


def _params(mock_urlopen: MagicMock, call: int = -1) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_urlopen.call_args_list[call].args[0].full_url).query)


def _failure(fn: Any, *args: Any, **kwargs: Any) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


class TestArxivSearch:
    @patch(HTTP_OPEN)
    def test_a_page_of_results_says_the_total_and_the_next_start(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            feed(entry("1706.03762v7"), entry("2301.00001v1"), total=1234, start=0)
        )

        result = arxiv_search("transformers", max_results=2)

        text = _text(result)
        assert text.startswith("arXiv papers that match 'transformers':\n1. Attention")
        assert "2. Attention" in text
        assert text.endswith("[results 1-2 of 1234 | next: start=2]")
        assert result.metadata["window"]["next_call"] == {"start": 2}
        params = _params(mock_urlopen)
        assert params["search_query"] == ['all:"transformers"']
        assert params["start"] == ["0"]
        assert params["max_results"] == ["2"]
        assert params["sortBy"] == ["relevance"]
        assert params["sortOrder"] == ["descending"]

    @patch(HTTP_OPEN)
    def test_the_next_page_is_numbered_on_and_the_last_says_end(self, mock_urlopen):
        mock_urlopen.return_value = respond(feed(entry("2301.00003v1"), total=3, start=2))

        text = _text(arxiv_search("transformers", max_results=2, start=2))

        assert text.splitlines()[1] == "3. Attention Is All You Need"
        assert text.endswith("[results 3-3 of 3 | end]")
        assert _params(mock_urlopen)["start"] == ["2"]

    @patch(HTTP_OPEN)
    def test_a_result_shows_the_whole_summary_and_says_how_many_authors_are_left(
        self, mock_urlopen
    ):
        summary = "A long abstract sentence. " * 60
        authors = tuple(f"Author {n}" for n in range(1, 13))
        mock_urlopen.return_value = respond(feed(entry(summary=summary, authors=authors), total=1))

        text = _text(arxiv_search("agents"))

        assert f"Summary: {summary.strip()}" in text  # whole: no field is cut below the window
        assert (
            "Authors: Author 1, Author 2, Author 3, Author 4, Author 5, Author 6, Author 7, "
            "Author 8 (+4 more; arxiv_paper('1706.03762v7') lists all)"
        ) in text
        assert "arXiv: 1706.03762v7 | cs.CL | published: 2017-06-12 | updated: 2023-08-02" in text
        assert "DOI: 10.48550/arXiv.1706.03762v7" in text
        assert "https://arxiv.org/abs/1706.03762v7 | https://arxiv.org/pdf/1706.03762v7" in text

    @patch(HTTP_OPEN)
    def test_zero_results_say_so_with_the_query(self, mock_urlopen):
        mock_urlopen.return_value = respond(_EMPTY)

        text = _text(arxiv_search("no such paper"))

        assert text == "No arXiv papers match 'no such paper'."

    @patch(HTTP_OPEN)
    def test_filters_category_dates_and_start(self, mock_urlopen):
        mock_urlopen.return_value = respond(_EMPTY)

        arxiv_search(
            "language agents",
            max_results=20,
            start=40,
            category="cs.AI",
            sort_by="submittedDate",
            from_date="2024-01-01",
            to_date="2024-01-31",
        )

        params = _params(mock_urlopen)
        assert params["max_results"] == ["20"]
        assert params["start"] == ["40"]
        assert params["sortBy"] == ["submittedDate"]
        assert params["search_query"] == [
            'cat:cs.AI AND all:"language agents" AND submittedDate:[202401010000 TO 202401312359]'
        ]

    @patch(HTTP_OPEN)
    def test_keeps_advanced_query_syntax(self, mock_urlopen):
        mock_urlopen.return_value = respond(_EMPTY)

        arxiv_search('ti:"language agents" AND cat:cs.AI')

        assert _params(mock_urlopen)["search_query"] == ['(ti:"language agents" AND cat:cs.AI)']

    @pytest.mark.parametrize(
        ("max_results", "start", "kept"),
        [(20, 29_999, True), (21, 0, False), (0, 0, False), (5, 30_000, False), (5, -1, False)],
    )
    @patch(HTTP_OPEN)
    def test_the_limits_are_the_schemas_and_the_executor_refuses_the_rest(
        self, mock_urlopen, max_results, start, kept
    ):
        # arXiv pages through the first 30,000 results; a page is at most 20 here.
        mock_urlopen.return_value = respond(_EMPTY)
        call = ToolCall(
            id="c1",
            name="arxiv_search",
            input={"query": "x", "max_results": max_results, "start": start},
        )

        result = ToolGroup(arxiv_search).execute(call)

        assert result.ok is kept
        if not kept:
            assert result.error is not None and result.error.type == "validation_error"

    @pytest.mark.parametrize(
        ("kwargs", "words"),
        [
            ({"query": "  "}, "query cannot be empty"),
            ({"query": "x", "sort_by": "date"}, "invalid sort_by 'date'"),
            ({"query": "x", "sort_order": "up"}, "invalid sort_order 'up'"),
            ({"query": "x", "category": "bad category"}, "invalid category"),
            ({"query": "x", "from_date": "01-01-2024"}, "invalid from_date"),
            (
                {"query": "x", "from_date": "2024-02-01", "to_date": "2024-01-01"},
                "must be before or equal to to_date",
            ),
        ],
    )
    @patch(HTTP_OPEN)
    def test_invalid_arguments_do_not_call_the_api(self, mock_urlopen, kwargs, words):
        failure = _failure(arxiv_search, **kwargs)

        assert failure.error.type == "validation_error"
        assert words in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_api_failure(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()

        failure = _failure(arxiv_search, "test")

        assert failure.error.type == "upstream"
        assert failure.error.retryable
        assert "timed out" in failure.error.message.lower()

    @patch(HTTP_OPEN)
    def test_parse_failure(self, mock_urlopen):
        mock_urlopen.return_value = respond("<not xml")

        failure = _failure(arxiv_search, "test")

        assert failure.error.type == "upstream"
        assert "could not parse" in failure.error.message

    def test_requests_are_three_seconds_apart_as_arxiv_asks(self):
        # https://info.arxiv.org/help/api/user-manual.html: "a 3 second delay" between calls.
        assert _arxiv._API.min_interval_s == 3.0


class TestArxivPaper:
    @patch(HTTP_OPEN)
    def test_the_record_is_whole_every_author_with_the_affiliation(self, mock_urlopen):
        summary = "A long abstract sentence. " * 60
        authors = tuple(f"Author {n}" for n in range(1, 13))
        mock_urlopen.return_value = respond(feed(entry(summary=summary, authors=authors), total=1))

        text = _text(arxiv_paper("https://arxiv.org/abs/1706.03762v7", max_chars=20_000))

        assert text.startswith("arXiv paper 1706.03762v7:\nAttention Is All You Need\n")
        assert f"Summary: {summary.strip()}" in text
        assert "Authors (12): Author 1 (Google Brain), Author 2, " in text
        assert "Author 12" in text
        assert "Categories: cs.CL (primary), cs.LG" in text
        assert "Comment: 15 pages, 5 figures" in text
        assert "Journal: NeurIPS 2017" in text
        assert "1. Attention" not in text
        assert "[chars" not in text  # nothing left out
        params = _params(mock_urlopen)
        assert params["id_list"] == ["1706.03762v7"]
        assert params["max_results"] == ["1"]

    @patch(HTTP_OPEN)
    def test_a_long_record_reads_on_through_the_window(self, mock_urlopen):
        authors = tuple(f"Collaborator number {n}" for n in range(1, 400))
        mock_urlopen.side_effect = [respond(feed(entry(authors=authors), total=1)) for _ in "ab"]

        first = arxiv_paper("1706.03762", max_chars=1000)
        second = arxiv_paper("1706.03762", max_chars=1000, offset=first.metadata["window"]["last"])

        window = first.metadata["window"]
        assert window["first"] == 0 and window["next_call"] == {"offset": window["last"]}
        assert _text(first).endswith(f"next: offset={window['last']}]")
        assert second.metadata["window"]["first"] == window["last"]
        assert "Summary:" in _text(first)  # the abstract comes before the long lists
        assert "Summary:" not in _text(second)

    @patch(HTTP_OPEN)
    def test_normalizes_pdf_url(self, mock_urlopen):
        mock_urlopen.return_value = respond(feed(entry(), total=1))

        arxiv_paper("https://arxiv.org/pdf/1706.03762v7.pdf")

        assert _params(mock_urlopen)["id_list"] == ["1706.03762v7"]

    @patch(HTTP_OPEN)
    def test_invalid_id(self, mock_urlopen):
        failure = _failure(arxiv_paper, "bad id")

        assert failure.error.type == "validation_error"
        assert "invalid arXiv ID" in failure.error.message
        mock_urlopen.assert_not_called()

    @pytest.mark.parametrize(
        "answer",
        [
            _EMPTY,
            # An entry with nothing in it is no paper either.
            feed(
                "<entry><id>http://arxiv.org/abs/2501.00000</id><title/><summary/></entry>",
                total=1,
            ),
        ],
    )
    @patch(HTTP_OPEN)
    def test_not_found(self, mock_urlopen, answer):
        mock_urlopen.return_value = respond(answer)

        failure = _failure(arxiv_paper, "2501.00000")

        assert failure.error.type == "not_found"
        assert "no arXiv paper with ID 2501.00000" in failure.error.message
        assert "arxiv_search" in failure.error.message


@patch(HTTP_OPEN)
def test_a_query_arxiv_cannot_read_is_explained(mock_urlopen):
    # It read as "HTTP error 400: Bad Request".
    mock_urlopen.side_effect = http_error(400, "Bad Request", body=ARXIV_ERROR_FEED)

    failure = _failure(arxiv_search, "ti:(")

    assert failure.error.type == "validation_error"
    assert not failure.error.retryable
    assert failure.error.details == {"status": 400}
    assert failure.error.message.startswith(
        "arXiv rejected the request: Invalid query string: '( ( )'; check"
    )


@patch(HTTP_OPEN)
def test_an_error_entry_is_the_error_not_a_paper(mock_urlopen):
    # The form the user manual documents, with HTTP 200, would have read as a paper titled
    # "Error".
    mock_urlopen.return_value = respond(
        ARXIV_ERROR_FEED.decode()
        .replace("https://arxiv.org/api/errors<", "http://arxiv.org/api/errors#bad_id<")
        .replace("Invalid query string: '( ( )'", "incorrect id format for 1234.1234")
    )

    failure = _failure(arxiv_paper, "1234.1234")

    assert failure.error.type == "validation_error"
    assert failure.error.message.startswith(
        "arXiv rejected the request: incorrect id format for 1234.1234; check"
    )


@patch(HTTP_OPEN)
def test_an_error_feed_with_a_server_status_is_retryable_upstream(mock_urlopen):
    mock_urlopen.side_effect = http_error(503, "Service Unavailable", body=ARXIV_ERROR_FEED)

    failure = _failure(arxiv_search, "agents")

    assert failure.error.type == "upstream"
    assert failure.error.retryable
    assert failure.error.message == "HTTP error 503: Invalid query string: '( ( )'"


@patch(HTTP_OPEN)
def test_a_404_is_endpoint_not_found(mock_urlopen):
    # The query endpoint answers an unknown ID with an empty feed, so a 404 means it moved.
    mock_urlopen.side_effect = http_error(404, "Not Found")

    failure = _failure(arxiv_paper, "1706.03762")

    assert failure.error.type == "upstream"
    assert "arXiv: endpoint not found (HTTP 404)" in failure.error.message


@patch(HTTP_OPEN)
def test_an_empty_page_inside_the_total_is_a_failure_to_retry_not_the_end(mock_urlopen):
    # arXiv's API sends an empty page at times; the footer would have said "end" at result 11.
    mock_urlopen.return_value = respond(feed(total=1234, start=10))

    failure = _failure(arxiv_search, "agents", start=10)

    assert failure.error.type == "upstream"
    assert failure.error.retryable
    assert "send the same call again" in failure.error.message


@patch(HTTP_OPEN)
def test_the_last_page_within_30000_asks_for_what_is_left(mock_urlopen):
    mock_urlopen.return_value = respond(feed(*(entry(f"x{n}") for n in range(5)), total=90_000))

    text = _text(arxiv_search("agents", max_results=20, start=29_995))

    assert _params(mock_urlopen)["max_results"] == ["5"]
    assert text.endswith("[results 29996-30000 of 90000 | the rest cannot be read here]")
