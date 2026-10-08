"""Tests for toolkit/tools/_pubmed.py (T06)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._pubmed import pubmed_article, pubmed_search
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond
from tests.toolkit.literature_answers import (
    EFETCH_ERROR,
    ESEARCH_ERROR,
    ESEARCH_NOTHING,
    NCBI_RATE_LIMITED,
    esearch,
    pubmed_set,
)
from tests.toolkit.literature_answers import pubmed_article as article_xml


def _text(result: ToolResult) -> str:
    assert isinstance(result, ToolResult) and result.ok, result
    assert isinstance(result.value, str)
    return result.value


def _sent(mock_urlopen: MagicMock, call: int) -> tuple[str, dict[str, list[str]]]:
    url = urlparse(mock_urlopen.call_args_list[call].args[0].full_url)
    return url.path, parse_qs(url.query)


def _failure(fn: Any, *args: Any, **kwargs: Any) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


class TestPubmedSearch:
    @patch(HTTP_OPEN)
    def test_a_page_says_the_total_and_the_next_start(self, mock_urlopen):
        mock_urlopen.side_effect = [
            respond(esearch(["1", "2"], count=4321)),
            respond(pubmed_set(article_xml("1"), article_xml("2"))),
        ]

        result = pubmed_search("deep learning", max_results=2)

        text = _text(result)
        assert text.splitlines()[:5] == [
            "PubMed articles that match 'deep learning':",
            "1. Deep learning",
            "   PMID: 1 | PMCID: PMC4567 | DOI: 10.1038/nature14539 | published: 2015-05-28",
            "   Authors: Yann LeCun, Yoshua Bengio",
            "   Journal: Nature",
        ]
        assert text.endswith("[results 1-2 of 4321 | next: start=2]")
        path, params = _sent(mock_urlopen, 0)
        assert path.endswith("/esearch.fcgi")
        assert params["term"] == ["deep learning"]
        assert params["retstart"] == ["0"]
        assert params["retmax"] == ["2"]
        assert params["sort"] == ["relevance"]
        path, params = _sent(mock_urlopen, 1)
        assert path.endswith("/efetch.fcgi")
        assert params["id"] == ["1,2"]

    @patch(HTTP_OPEN)
    def test_long_author_lists_say_how_many_more_and_where(self, mock_urlopen):
        mock_urlopen.side_effect = [
            respond(esearch(["1"], count=1)),
            respond(pubmed_set(article_xml("1", authors=11))),
        ]

        text = _text(pubmed_search("deep learning"))

        assert "(+3 more; pubmed_article('1') lists all)" in text
        assert "MeSH" not in text  # the record lists them all

    @patch(HTTP_OPEN)
    def test_an_article_efetch_does_not_return_keeps_its_place(self, mock_urlopen):
        mock_urlopen.side_effect = [
            respond(esearch(["1", "2"], count=2)),
            respond(pubmed_set(article_xml("2"))),
        ]

        text = _text(pubmed_search("deep learning", max_results=2))

        assert "1. PMID 1: PubMed returned no record for it; try pubmed_article('1')" in text
        assert "2. Deep learning" in text

    @patch(HTTP_OPEN)
    def test_esearch_reaches_the_first_10000_records(self, mock_urlopen):
        # retstart + retmax <= 10,000 (https://www.nlm.nih.gov/pubs/techbull/so22/
        # so22_updated_pubmed_e_utilities.html): the last page asks for what is left.
        mock_urlopen.side_effect = [
            respond(esearch(["1", "2"], count=50_000, start=9_998)),
            respond(pubmed_set(article_xml("1"), article_xml("2"))),
        ]

        text = _text(pubmed_search("deep learning", max_results=5, start=9_998))

        assert _sent(mock_urlopen, 0)[1]["retmax"] == ["2"]
        assert text.endswith("[results 9999-10000 of 50000 | the rest cannot be read here]")

    @patch(HTTP_OPEN)
    def test_zero_results_say_so_with_the_query_and_what_pubmed_did_not_find(self, mock_urlopen):
        mock_urlopen.return_value = respond(ESEARCH_NOTHING)

        text = _text(pubmed_search('"zzqqxxnotaword"'))

        assert text == (
            'No PubMed articles match \'"zzqqxxnotaword"\'. PubMed did not find: "zzqqxxnotaword".'
        )
        assert mock_urlopen.call_count == 1

    @patch(HTTP_OPEN)
    def test_filters_dates_sort_and_start(self, mock_urlopen):
        mock_urlopen.return_value = respond(esearch([], count=0, start=40))

        pubmed_search(
            "deep learning",
            max_results=20,
            start=40,
            from_date="2024-01-01",
            to_date="2024-01-31",
            sort="pub_date",
        )

        params = _sent(mock_urlopen, 0)[1]
        assert params["retmax"] == ["20"]
        assert params["retstart"] == ["40"]
        assert params["sort"] == ["pub date"]
        assert params["datetype"] == ["pdat"]
        assert params["mindate"] == ["2024/01/01"]
        assert params["maxdate"] == ["2024/01/31"]

    @pytest.mark.parametrize(
        ("max_results", "start", "kept"),
        [(20, 9_999, True), (21, 0, False), (0, 0, False), (5, 10_000, False)],
    )
    @patch(HTTP_OPEN)
    def test_the_limits_are_the_schemas(self, mock_urlopen, max_results, start, kept):
        mock_urlopen.return_value = respond(esearch([], count=0))
        call = ToolCall(
            id="c1",
            name="pubmed_search",
            input={"query": "x", "max_results": max_results, "start": start},
        )

        assert ToolGroup(pubmed_search).execute(call).ok is kept

    @pytest.mark.parametrize(
        ("kwargs", "words"),
        [
            ({"query": " "}, "query cannot be empty"),
            ({"query": "x", "sort": "date"}, "unknown sort 'date'"),
            ({"query": "x", "from_date": "2024/01/01"}, "invalid from_date"),
            ({"query": "x", "from_date": "2024-02-01", "to_date": "2024-01-01"}, "is after"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen, kwargs, words):
        failure = _failure(pubmed_search, **kwargs)

        assert failure.error.type == "validation_error"
        assert words in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_an_error_the_search_reports_is_the_tools_error(self, mock_urlopen):
        # Sent with HTTP 200 (eutils.ncbi.nlm.nih.gov, 2026-09-29); it used to read as no results.
        mock_urlopen.return_value = respond(ESEARCH_ERROR)

        failure = _failure(pubmed_search, "(((")

        assert failure.error.type == "upstream"
        assert failure.error.message == (
            "Search Backend failed: An error occurred while processing "
            "request. Details: Empty Term in the request"
        )
        assert mock_urlopen.call_count == 1

    @patch(HTTP_OPEN)
    def test_a_404_on_the_search_is_an_endpoint_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(pubmed_search, "test")

        assert failure.error.type == "upstream"
        assert "NCBI E-utilities: endpoint not found (HTTP 404)" in failure.error.message

    @patch(HTTP_OPEN)
    def test_ncbi_s_rate_limit_keeps_its_words(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(429, "Too Many Requests", body=NCBI_RATE_LIMITED)

        failure = _failure(pubmed_search, "test")

        assert failure.error.type == "rate_limited"
        assert failure.error.retryable
        assert "API rate limit exceeded" in failure.error.message

    @patch(HTTP_OPEN)
    def test_article_xml_parse_failure(self, mock_urlopen):
        mock_urlopen.side_effect = [respond(esearch(["1"], count=1)), respond("<not xml")]

        failure = _failure(pubmed_search, "test")

        assert failure.error.type == "upstream"
        assert "could not parse" in failure.error.message


class TestPubmedArticle:
    @patch(HTTP_OPEN)
    def test_the_record_is_whole(self, mock_urlopen):
        mock_urlopen.return_value = respond(pubmed_set(article_xml(authors=11)))

        text = _text(pubmed_article("26017442"))

        lines = text.splitlines()
        assert lines[:7] == [
            "PubMed article 26017442:",
            "Deep learning",
            "PMID: 26017442 | PMCID: PMC4567 | DOI: 10.1038/nature14539 | published: 2015-05-28",
            "Journal: Nature | Publication types: Journal Article, Review",
            "Abstract:",
            "BACKGROUND: Deep learning allows computational models.",
            "CONCLUSIONS: It is useful in many domains.",
        ]
        assert "Authors (11): Yann LeCun (Facebook AI Research), Yoshua Bengio, " in text
        assert "Given Author11" in text
        assert "MeSH: Machine Learning (major; methods), Neural Networks, Computer" in text
        assert "Keywords: deep learning, neural networks" in text
        assert text.endswith("URL: https://pubmed.ncbi.nlm.nih.gov/26017442/\n")
        params = _sent(mock_urlopen, 0)[1]
        assert params["id"] == ["26017442"]
        assert params["retmode"] == ["xml"]

    @patch(HTTP_OPEN)
    def test_a_long_abstract_reads_on_through_the_window(self, mock_urlopen):
        long = "A sentence of the abstract. " * 100
        mock_urlopen.side_effect = [
            respond(pubmed_set(article_xml(abstract=long))) for _ in range(2)
        ]

        first = pubmed_article("26017442", max_chars=1000)
        last = first.metadata["window"]["last"]
        second = pubmed_article("26017442", max_chars=1000, offset=last)

        assert _text(first).endswith(f"next: offset={last}]")
        assert second.metadata["window"]["first"] == last

    @patch(HTTP_OPEN)
    def test_invalid_pmid(self, mock_urlopen):
        failure = _failure(pubmed_article, "PMID 1")

        assert failure.error.type == "validation_error"
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_not_found(self, mock_urlopen):
        mock_urlopen.return_value = respond(pubmed_set())

        failure = _failure(pubmed_article, "99999999")

        assert failure.error.type == "not_found"
        assert "pubmed_search" in failure.error.message

    @patch(HTTP_OPEN)
    def test_an_error_efetch_reports_is_the_error_not_a_missing_article(self, mock_urlopen):
        mock_urlopen.return_value = respond(EFETCH_ERROR)

        failure = _failure(pubmed_article, "26017442")

        assert failure.error.type == "upstream"
        assert failure.error.retryable
        assert failure.error.message == (
            "PubMed EFetch error: Unable to obtain query #1; try again later, or check the "
            "PMID with pubmed_search"
        )
