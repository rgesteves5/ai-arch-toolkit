"""Tests for toolkit/tools/_europe_pmc.py."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._europe_pmc import (
    europe_pmc_article,
    europe_pmc_citations,
    europe_pmc_search,
)

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


def _mock_urlopen(data: dict | str):
    resp = MagicMock()
    if isinstance(data, dict):
        resp.read.return_value = json.dumps(data).encode()
    else:
        resp.read.return_value = data.encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _called_request(mock_urlopen):
    return mock_urlopen.call_args.args[0]


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    return parse_qs(urlparse(_called_request(mock_urlopen).full_url).query)


class TestEuropePmcSearch:
    @patch("ai_arch_toolkit.toolkit.tools._europe_pmc.urllib.request.urlopen")
    def test_returns_search_results(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_SEARCH)

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

    @patch("ai_arch_toolkit.toolkit.tools._europe_pmc.urllib.request.urlopen")
    def test_invalid_search_options_do_not_call_api(self, mock_urlopen):
        assert "query cannot be empty" in europe_pmc_search("")
        assert "result_type must" in europe_pmc_search("test", result_type="full")
        mock_urlopen.assert_not_called()


class TestEuropePmcArticle:
    @patch("ai_arch_toolkit.toolkit.tools._europe_pmc.urllib.request.urlopen")
    def test_returns_article_by_pmid_and_source(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_SEARCH)

        result = europe_pmc_article("26017442", source="MED")

        assert result.startswith("Europe PMC article MED/26017442:")
        assert "Abstract: Deep learning allows computational models." in result
        assert "Full text: https://example.test/fulltext" in result

        params = _called_params(mock_urlopen)
        assert params["query"] == ["SRC:MED AND EXT_ID:26017442"]
        assert params["resultType"] == ["core"]

    @patch("ai_arch_toolkit.toolkit.tools._europe_pmc.urllib.request.urlopen")
    def test_article_query_by_doi(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_SEARCH)

        europe_pmc_article("10.1038/nature14539")

        assert _called_params(mock_urlopen)["query"] == ['DOI:"10.1038/nature14539"']

    @patch("ai_arch_toolkit.toolkit.tools._europe_pmc.urllib.request.urlopen")
    def test_invalid_article_options_do_not_call_api(self, mock_urlopen):
        assert "identifier cannot be empty" in europe_pmc_article("")
        assert "invalid source" in europe_pmc_article("26017442", source="bad!")
        mock_urlopen.assert_not_called()


class TestEuropePmcCitations:
    @patch("ai_arch_toolkit.toolkit.tools._europe_pmc.urllib.request.urlopen")
    def test_returns_citations(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
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

    @patch("ai_arch_toolkit.toolkit.tools._europe_pmc.urllib.request.urlopen")
    def test_citations_no_results(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
            {"hitCount": 0, "citationList": {"citation": []}}
        )

        result = europe_pmc_citations("MED", "missing")

        assert "No Europe PMC citations found" in result

    @patch("ai_arch_toolkit.toolkit.tools._europe_pmc.urllib.request.urlopen")
    def test_api_and_parse_failures(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )
        assert "rate limited" in europe_pmc_search("test")

        mock_urlopen.side_effect = None
        mock_urlopen.return_value = _mock_urlopen("not json")
        assert "could not parse" in europe_pmc_search("test")
