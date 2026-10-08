"""Tests for toolkit/tools/_europe_pmc.py (T06)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._europe_pmc import (
    europe_pmc_article,
    europe_pmc_citations,
    europe_pmc_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond
from tests.toolkit.literature_answers import (
    EPMC_ERROR,
    epmc_citation,
    epmc_citations,
    epmc_result,
    epmc_search,
)


def _text(result: ToolResult) -> str:
    assert isinstance(result, ToolResult) and result.ok, result
    assert isinstance(result.value, str)
    return result.value


def _sent(mock_urlopen: MagicMock, call: int = -1) -> tuple[str, dict[str, list[str]]]:
    url = urlparse(mock_urlopen.call_args_list[call].args[0].full_url)
    return url.path, parse_qs(url.query)


def _failure(fn: Any, *args: Any, **kwargs: Any) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


class TestEuropePmcSearch:
    @patch(HTTP_OPEN)
    def test_a_page_says_the_total_and_the_next_cursor_with_the_position(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            epmc_search(epmc_result("1"), epmc_result("2"), hit_count=57, next_cursor="AoJ")
        )

        result = europe_pmc_search("deep learning", max_results=2, result_type="core")

        text = _text(result)
        assert text.splitlines()[:6] == [
            "Europe PMC articles that match 'deep learning':",
            "1. Deep learning",
            "   id: MED/1 | PMID: 1 | PMCID: PMC123 | DOI: 10.1038/nature14539 | year: 2015 | "
            "cited by: 123",
            "   open access: yes | in Europe PMC: yes | in PMC: no | PDF: yes | references: yes",
            "   Authors: LeCun Y, Bengio Y, Hinton G",
            "   Journal: Nature | Type: Review; Journal Article",
        ]
        assert text.endswith('[results 1-2 of 57 | next: cursor_mark="AoJ", offset=2]')
        path, params = _sent(mock_urlopen)
        assert path.endswith("/search")
        assert params["query"] == ["deep learning"]
        assert params["pageSize"] == ["2"]
        assert params["cursorMark"] == ["*"]
        assert params["resultType"] == ["core"]

    @patch(HTTP_OPEN)
    def test_the_next_page_is_numbered_from_the_offset(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            epmc_search(epmc_result("3"), hit_count=3, cursor="AoJ", next_cursor="AoK")
        )

        text = _text(
            europe_pmc_search("deep learning", max_results=2, cursor_mark="AoJ", offset=2)
        )

        assert text.splitlines()[1] == "3. Deep learning"
        assert text.endswith("[results 3-3 of 3 | end]")
        assert _sent(mock_urlopen)[1]["cursorMark"] == ["AoJ"]

    @patch(HTTP_OPEN)
    def test_a_long_author_list_says_how_many_more_and_where(self, mock_urlopen):
        mock_urlopen.return_value = respond(epmc_search(epmc_result(authors=12), hit_count=1))

        text = _text(europe_pmc_search("deep learning"))

        assert "(+4 more; europe_pmc_article('MED/26017442') lists all)" in text

    @patch(HTTP_OPEN)
    def test_zero_results_say_so_with_the_query(self, mock_urlopen):
        mock_urlopen.return_value = respond(epmc_search(hit_count=0))

        assert _text(europe_pmc_search("zzqq")) == "No Europe PMC articles match 'zzqq'."

    @patch(HTTP_OPEN)
    def test_an_error_in_place_of_the_result_is_the_error(self, mock_urlopen):
        mock_urlopen.return_value = respond(EPMC_ERROR)

        failure = _failure(europe_pmc_search, "deep learning")

        assert failure.error.type == "upstream"
        assert failure.error.message == (
            "Europe PMC error 404: Invalid page size provided. Valid size is between 1 and 1000; "
            "check the query and the identifiers, or try again later"
        )

    @pytest.mark.parametrize(
        ("max_results", "offset", "kept"), [(20, 0, True), (21, 0, False), (5, -1, False)]
    )
    @patch(HTTP_OPEN)
    def test_the_limits_are_the_schemas(self, mock_urlopen, max_results, offset, kept):
        mock_urlopen.return_value = respond(epmc_search(hit_count=0))
        call = ToolCall(
            id="c1",
            name="europe_pmc_search",
            input={"query": "x", "max_results": max_results, "offset": offset},
        )

        assert ToolGroup(europe_pmc_search).execute(call).ok is kept

    @patch(HTTP_OPEN)
    def test_invalid_arguments(self, mock_urlopen):
        assert "query cannot be empty" in _failure(europe_pmc_search, " ").error.message
        bad = _failure(europe_pmc_search, "x", result_type="full")
        assert "result_type must be one of" in bad.error.message
        mock_urlopen.assert_not_called()


class TestEuropePmcArticle:
    @patch(HTTP_OPEN)
    def test_the_record_is_whole(self, mock_urlopen):
        mock_urlopen.return_value = respond(epmc_search(epmc_result(authors=12), hit_count=1))

        text = _text(europe_pmc_article("26017442"))

        lines = text.splitlines()
        assert lines[:6] == [
            "Europe PMC article MED/26017442:",
            "Deep learning",
            "id: MED/26017442 | PMID: 26017442 | PMCID: PMC123 | DOI: 10.1038/nature14539 | "
            "year: 2015 | cited by: 123",
            "published: 2015-05-28 | open access: yes | in Europe PMC: yes | in PMC: no | "
            "PDF: yes | references: yes",
            "Journal: Nature | Type: Review; Journal Article",
            "Abstract: Deep learning allows computational models.",
        ]
        assert "Authors (12): LeCun Y (NYU), Bengio Y, Hinton G, Author4 A, " in text
        assert "Author12 A" in text
        assert "MeSH: Neural Networks, Computer (major; trends), Humans" in text
        assert "Keywords: deep learning, representation" in text
        assert (
            "Full text: https://doi.org/10.1038/nature14539 (Subscription required, doi) | "
            "https://europepmc.org/articles/PMC123?pdf=render (Open access, pdf)"
        ) in text
        assert text.endswith("Europe PMC: https://europepmc.org/article/MED/26017442\n")
        _path, params = _sent(mock_urlopen)
        assert params["query"] == ["EXT_ID:26017442"]
        assert params["resultType"] == ["core"]

    @pytest.mark.parametrize(
        ("identifier", "source", "query"),
        [
            ("PMC123", "", "PMCID:PMC123"),
            ("10.1038/nature14539", "", 'DOI:"10.1038/nature14539"'),
            ("123", "med", "SRC:MED AND EXT_ID:123"),
            ("MED/26017442", "", "SRC:MED AND EXT_ID:26017442"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_the_identifier_forms_it_takes(self, mock_urlopen, identifier, source, query):
        mock_urlopen.return_value = respond(epmc_search(epmc_result(), hit_count=1))

        europe_pmc_article(identifier, source=source)

        assert _sent(mock_urlopen)[1]["query"] == [query]

    @patch(HTTP_OPEN)
    def test_an_identifier_several_sources_share_says_which_one_it_shows(self, mock_urlopen):
        mock_urlopen.return_value = respond(epmc_search(epmc_result(), hit_count=2))

        text = _text(europe_pmc_article("26017442"))

        assert text.startswith(
            "Europe PMC article MED/26017442 (the first of 2 records with ID '26017442'; "
            "give source= for another):"
        )

    @patch(HTTP_OPEN)
    def test_a_long_abstract_reads_on_through_the_window(self, mock_urlopen):
        long = "<p>" + "A sentence of the abstract. " * 100 + "</p>"
        mock_urlopen.side_effect = [
            respond(epmc_search(epmc_result(abstract=long), hit_count=1)) for _ in range(2)
        ]

        first = europe_pmc_article("26017442", max_chars=1000)
        last = first.metadata["window"]["last"]
        second = europe_pmc_article("26017442", max_chars=1000, offset=last)

        assert _text(first).endswith(f"next: offset={last}]")
        assert second.metadata["window"]["first"] == last

    @patch(HTTP_OPEN)
    def test_not_found(self, mock_urlopen):
        mock_urlopen.return_value = respond(epmc_search(hit_count=0))

        failure = _failure(europe_pmc_article, "99999999")

        assert failure.error.type == "not_found"
        assert "europe_pmc_search" in failure.error.message

    @patch(HTTP_OPEN)
    def test_invalid_arguments(self, mock_urlopen):
        assert "identifier cannot be empty" in _failure(europe_pmc_article, " ").error.message
        assert "invalid source" in _failure(europe_pmc_article, "1", source="MEDX").error.message
        mock_urlopen.assert_not_called()


class TestEuropePmcCitations:
    @patch(HTTP_OPEN)
    def test_a_page_of_citations_says_the_total_and_the_next_page(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            epmc_citations(epmc_citation("1"), epmc_citation("2"), hit_count=57)
        )

        result = europe_pmc_citations("med", "26017442", max_results=2, page=3)

        text = _text(result)
        assert text.splitlines()[:5] == [
            "Articles in Europe PMC that cite MED/26017442:",
            "5. Quantum Machine Learning",
            "   id: MED/1 | year: 2026 | cited by: 0",
            "   Authors: Liu H, Chen J",
            "   Journal: Ann Biomed Eng | Type: journal article",
        ]
        assert text.endswith("[results 5-6 of 57 | next: page=4, max_results=2]")
        path, params = _sent(mock_urlopen)
        assert path.endswith("/MED/26017442/citations")
        assert params["page"] == ["3"]
        assert params["pageSize"] == ["2"]
        assert params["format"] == ["json"]

    @patch(HTTP_OPEN)
    def test_a_record_nobody_cites_says_so(self, mock_urlopen):
        mock_urlopen.side_effect = [
            respond(epmc_citations(hit_count=0)),
            respond(epmc_search(epmc_result(), hit_count=1)),
        ]

        text = _text(europe_pmc_citations("MED", "26017442"))

        assert text == "No articles in Europe PMC cite MED/26017442."
        _path, params = _sent(mock_urlopen)
        assert params["query"] == ["SRC:MED AND EXT_ID:26017442"]

    @patch(HTTP_OPEN)
    def test_a_record_europe_pmc_does_not_have_is_not_found(self, mock_urlopen):
        # The citations endpoint answers an unknown record as one nobody cites, so the record is
        # looked up before saying so.
        mock_urlopen.side_effect = [
            respond(epmc_citations(hit_count=0)),
            respond(epmc_search(hit_count=0)),
        ]

        failure = _failure(europe_pmc_citations, "MED", "99999999")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "Europe PMC has no record MED/99999999; find the article's source and ID with "
            "europe_pmc_search"
        )

    @pytest.mark.parametrize(
        ("max_results", "page", "kept"), [(25, 1, True), (26, 1, False), (5, 0, False)]
    )
    @patch(HTTP_OPEN)
    def test_the_limits_are_the_schemas(self, mock_urlopen, max_results, page, kept):
        mock_urlopen.return_value = respond(epmc_citations(epmc_citation(), hit_count=1))
        call = ToolCall(
            id="c1",
            name="europe_pmc_citations",
            input={"source": "MED", "identifier": "1", "max_results": max_results, "page": page},
        )

        assert ToolGroup(europe_pmc_citations).execute(call).ok is kept

    @patch(HTTP_OPEN)
    def test_invalid_arguments(self, mock_urlopen):
        assert "invalid source" in _failure(europe_pmc_citations, "M1", "1").error.message
        empty = _failure(europe_pmc_citations, "MED", " ")
        assert "identifier cannot be empty" in empty.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_server_errors_are_upstream(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(503, "Service Unavailable")

        failure = _failure(europe_pmc_citations, "MED", "1")

        assert failure.error.type == "upstream"
        assert failure.error.retryable
