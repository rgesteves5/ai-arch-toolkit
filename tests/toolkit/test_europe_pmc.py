"""Tests for toolkit/tools/_europe_pmc.py."""

from __future__ import annotations

import urllib.error
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._europe_pmc import (
    europe_pmc_article,
    europe_pmc_citations,
    europe_pmc_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, respond

_ARTICLE = {
    "id": "26017442",
    "source": "MED",
    "pmid": "26017442",
    "pmcid": "PMC123",
    "doi": "10.1038/nature14539",
    "title": "Deep <i>learning</i>",
    "authorString": "LeCun Y, Bengio Y, Hinton G.",
    "journalTitle": "Nature",
    "pubYear": "2015",
    "firstPublicationDate": "2015-05-28",
    "pubType": "journal article",
    "abstractText": "<p>Deep learning allows computational models.</p>",
    "isOpenAccess": "Y",
    "inEPMC": "Y",
    "inPMC": "N",
    "hasPDF": "Y",
    "hasReferences": "Y",
    "citedByCount": 123,
    "fullTextUrlList": {"fullTextUrl": [{"url": "https://example.test/fulltext"}]},
}
_SEARCH = {
    "hitCount": 1,
    "nextCursorMark": "next",
    "resultList": {"result": [_ARTICLE]},
}
_CITATION = {
    "id": "42207361",
    "source": "MED",
    "citationType": "journal article",
    "title": "Quantum Machine Learning",
    "authorString": "Liu H.",
    "journalAbbreviation": "Ann Biomed Eng",
    "pubYear": 2026,
    "citedByCount": 0,
}


def _failure(call) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value


def _called_request(mock_urlopen):
    return mock_urlopen.call_args.args[0]


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(_called_request(mock_urlopen).full_url).query)


class TestEuropePmcSearch:
    @patch(HTTP_OPEN)
    def test_returns_search_results(self, mock_urlopen):
        mock_urlopen.return_value = respond(_SEARCH)

        result = europe_pmc_search(
            "deep learning", max_results=2, cursor_mark="*", result_type="core"
        )

        assert "Europe PMC results for 'deep learning' (total 1) | nextCursorMark: next" in result
        assert "Deep learning" in result
        assert "PMID: 26017442" in result
        assert "DOI: 10.1038/nature14539" in result
        assert "open access: Y" in result
        assert "cited by: 123" in result
        assert "Abstract:" not in result

        request = _called_request(mock_urlopen)
        assert request.headers["User-agent"].startswith("ai-arch-toolkit/")
        params = _called_params(mock_urlopen)
        assert params["query"] == ["deep learning"]
        assert params["pageSize"] == ["2"]
        assert params["cursorMark"] == ["*"]
        assert params["resultType"] == ["core"]

    @patch(HTTP_OPEN)
    def test_invalid_search_options_do_not_call_api(self, mock_urlopen):
        for call, words in (
            (lambda: europe_pmc_search(""), "query cannot be empty"),
            (lambda: europe_pmc_search("test", result_type="full"), "result_type must"),
        ):
            failure = _failure(call)
            assert failure.error.type == "validation_error"
            assert words in str(failure)
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_no_results_is_a_success(self, mock_urlopen):
        mock_urlopen.return_value = respond({"hitCount": 0, "resultList": {"result": []}})

        assert europe_pmc_search("zzz") == "No Europe PMC results for: 'zzz'"


class TestEuropePmcArticle:
    @patch(HTTP_OPEN)
    def test_returns_article_by_pmid_and_source(self, mock_urlopen):
        mock_urlopen.return_value = respond(_SEARCH)

        result = europe_pmc_article("26017442", source="MED")

        assert result.startswith("Europe PMC article MED/26017442:")
        assert "Abstract: Deep learning allows computational models." in result
        assert "Full text: https://example.test/fulltext" in result

        params = _called_params(mock_urlopen)
        assert params["query"] == ["SRC:MED AND EXT_ID:26017442"]
        assert params["resultType"] == ["core"]

    @patch(HTTP_OPEN)
    def test_article_query_by_doi(self, mock_urlopen):
        mock_urlopen.return_value = respond(_SEARCH)

        europe_pmc_article("10.1038/nature14539")

        assert _called_params(mock_urlopen)["query"] == ['DOI:"10.1038/nature14539"']

    @patch(HTTP_OPEN)
    def test_invalid_article_options_do_not_call_api(self, mock_urlopen):
        for call, words in (
            (lambda: europe_pmc_article(""), "identifier cannot be empty"),
            (lambda: europe_pmc_article("26017442", source="bad!"), "invalid source"),
        ):
            failure = _failure(call)
            assert failure.error.type == "validation_error"
            assert words in str(failure)
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_an_unknown_article_is_not_found(self, mock_urlopen):
        mock_urlopen.return_value = respond({"hitCount": 0, "resultList": {"result": []}})

        failure = _failure(lambda: europe_pmc_article("99999999"))

        assert failure.error.type == "not_found"
        assert "'99999999'" in str(failure)
        assert "europe_pmc_search" in str(failure)


class TestEuropePmcCitations:
    @patch(HTTP_OPEN)
    def test_returns_citations(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {"hitCount": 1, "citationList": {"citation": [_CITATION]}}
        )

        result = europe_pmc_citations("MED", "26017442", max_results=2)

        assert "Europe PMC citations for MED/26017442" in result
        assert "Quantum Machine Learning" in result
        assert "id: MED/42207361" in result

        request = _called_request(mock_urlopen)
        assert (
            urlparse(request.full_url).path == "/europepmc/webservices/rest/MED/26017442/citations"
        )
        assert _called_params(mock_urlopen)["pageSize"] == ["2"]

    @patch(HTTP_OPEN)
    def test_citations_no_results(self, mock_urlopen):
        mock_urlopen.return_value = respond({"hitCount": 0, "citationList": {"citation": []}})

        result = europe_pmc_citations("MED", "missing")

        assert "No Europe PMC citations found" in result

    @patch(HTTP_OPEN)
    def test_invalid_citation_options_do_not_call_api(self, mock_urlopen):
        for call, words in (
            (lambda: europe_pmc_citations("bad!", "1"), "invalid source"),
            (lambda: europe_pmc_citations("MED", " "), "identifier cannot be empty"),
        ):
            failure = _failure(call)
            assert failure.error.type == "validation_error"
            assert words in str(failure)
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_api_and_parse_failures(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )
        failure = _failure(lambda: europe_pmc_search("test"))
        assert failure.error.type == "rate_limited"
        assert "rate limited" in str(failure)

        mock_urlopen.side_effect = None
        mock_urlopen.return_value = respond("not json")
        failure = _failure(lambda: europe_pmc_search("test"))
        assert failure.error.type == "upstream"
        assert "could not parse" in str(failure)
