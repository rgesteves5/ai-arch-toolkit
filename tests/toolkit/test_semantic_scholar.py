"""Tests for toolkit/tools/_semantic_scholar.py."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._semantic_scholar import (
    semantic_scholar_citations,
    semantic_scholar_paper,
    semantic_scholar_search,
)

_PAPER = {
    "paperId": "649def34f8be52c8b66281af98ae884c09aef38b",
    "corpusId": 13756489,
    "title": "Attention Is All You Need",
    "abstract": "The dominant sequence transduction models are based on recurrence.",
    "year": 2017,
    "venue": "NeurIPS",
    "publicationVenue": {"name": "Neural Information Processing Systems"},
    "publicationTypes": ["Conference"],
    "publicationDate": "2017-06-12",
    "url": "https://www.semanticscholar.org/paper/example",
    "externalIds": {
        "DOI": "10.48550/arXiv.1706.03762",
        "ArXiv": "1706.03762",
        "CorpusId": 13756489,
    },
    "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}],
    "citationCount": 123456,
    "referenceCount": 50,
    "influentialCitationCount": 25000,
    "openAccessPdf": {"url": "https://arxiv.org/pdf/1706.03762"},
    "fieldsOfStudy": ["Computer Science"],
    "s2FieldsOfStudy": [{"category": "Machine Learning"}],
}

_SEARCH_RESPONSE = {"data": [_PAPER]}
_EMPTY_RESPONSE = {"data": []}
_CITATION_RESPONSE = {
    "data": [
        {
            "contexts": ["This work follows the Transformer architecture."],
            "intents": ["background", "methodology"],
            "isInfluential": True,
            "citingPaper": {
                **_PAPER,
                "paperId": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "year": 2019,
                "citationCount": 90000,
            },
        }
    ]
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


class TestSemanticScholarSearch:
    @patch("ai_arch_toolkit.toolkit.tools._semantic_scholar.urllib.request.urlopen")
    def test_returns_results(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_SEARCH_RESPONSE)

        result = semantic_scholar_search("attention", max_results=2)

        assert "Semantic Scholar results for 'attention'" in result
        assert "Attention Is All You Need" in result
        assert "paperId: 649def34f8be52c8b66281af98ae884c09aef38b" in result
        assert "year: 2017 | published: 2017-06-12" in result
        assert "Ashish Vaswani, Noam Shazeer" in result
        assert "Venue: Neural Information Processing Systems" in result
        assert "Citations: 123456" in result
        assert "DOI: 10.48550/arXiv.1706.03762" in result
        assert "Open PDF: https://arxiv.org/pdf/1706.03762" in result
        assert "Abstract:" not in result

        params = _called_params(mock_urlopen)
        assert params["query"] == ["attention"]
        assert params["limit"] == ["2"]
        assert params["offset"] == ["0"]
        assert "paperId" in params["fields"][0]

    @patch("ai_arch_toolkit.toolkit.tools._semantic_scholar.urllib.request.urlopen")
    def test_filters_year_venue_start_and_caps_max_results(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_EMPTY_RESPONSE)

        semantic_scholar_search(
            "language agents", max_results=99, start=40, year="2024", venue="ACL"
        )

        params = _called_params(mock_urlopen)
        assert params["query"] == ["language agents"]
        assert params["limit"] == ["20"]
        assert params["offset"] == ["40"]
        assert params["year"] == ["2024"]
        assert params["venue"] == ["ACL"]

    @patch("ai_arch_toolkit.toolkit.tools._semantic_scholar.urllib.request.urlopen")
    def test_no_results(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_EMPTY_RESPONSE)

        result = semantic_scholar_search("no such paper")

        assert "No Semantic Scholar results" in result

    @patch("ai_arch_toolkit.toolkit.tools._semantic_scholar.urllib.request.urlopen")
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "query cannot be empty" in semantic_scholar_search("")
        assert "start must be greater than or equal to 0" in semantic_scholar_search(
            "test", start=-1
        )
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._semantic_scholar.urllib.request.urlopen")
    def test_rate_limit(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.semanticscholar.org/graph/v1/paper/search",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )

        result = semantic_scholar_search("test")

        assert "rate limited" in result

    @patch("ai_arch_toolkit.toolkit.tools._semantic_scholar.urllib.request.urlopen")
    def test_parse_failure(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen("not json")

        result = semantic_scholar_search("test")

        assert "could not parse" in result


class TestSemanticScholarPaper:
    @patch("ai_arch_toolkit.toolkit.tools._semantic_scholar.urllib.request.urlopen")
    def test_returns_paper_by_doi_url(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_PAPER)

        result = semantic_scholar_paper("https://doi.org/10.48550/arXiv.1706.03762")

        assert result.startswith("Semantic Scholar paper DOI:10.48550/arXiv.1706.03762:")
        assert "Attention Is All You Need" in result
        assert "Abstract: The dominant sequence transduction models" in result
        assert "Fields: Computer Science, Machine Learning" in result
        assert "Publication types: Conference" in result
        assert "1. Attention" not in result

        request = _called_request(mock_urlopen)
        assert urlparse(request.full_url).path.endswith("/DOI:10.48550%2FarXiv.1706.03762")

    @patch("ai_arch_toolkit.toolkit.tools._semantic_scholar.urllib.request.urlopen")
    def test_normalizes_arxiv_url_and_pmid(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_PAPER)

        semantic_scholar_paper("https://arxiv.org/pdf/1706.03762v7.pdf")
        assert urlparse(_called_request(mock_urlopen).full_url).path.endswith(
            "/ARXIV:1706.03762v7"
        )

        semantic_scholar_paper("26017442")
        assert urlparse(_called_request(mock_urlopen).full_url).path.endswith("/PMID:26017442")

    @patch("ai_arch_toolkit.toolkit.tools._semantic_scholar.urllib.request.urlopen")
    def test_invalid_paper_id(self, mock_urlopen):
        result = semantic_scholar_paper("")

        assert "invalid paper_id" in result
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._semantic_scholar.urllib.request.urlopen")
    def test_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.semanticscholar.org/graph/v1/paper/missing",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

        result = semantic_scholar_paper("missing")

        assert "not found" in result.lower()


class TestSemanticScholarCitations:
    @patch("ai_arch_toolkit.toolkit.tools._semantic_scholar.urllib.request.urlopen")
    def test_returns_citations(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_CITATION_RESPONSE)

        result = semantic_scholar_citations("ARXIV:1706.03762", max_results=2, start=5)

        assert "Semantic Scholar citations for ARXIV:1706.03762" in result
        assert "BERT: Pre-training" in result
        assert "Citation: influential | intents: background, methodology" in result
        assert "Context: This work follows the Transformer architecture." in result

        params = _called_params(mock_urlopen)
        assert params["limit"] == ["2"]
        assert params["offset"] == ["5"]
        assert "citingPaper.title" in params["fields"][0]

    @patch("ai_arch_toolkit.toolkit.tools._semantic_scholar.urllib.request.urlopen")
    def test_rejects_negative_start(self, mock_urlopen):
        result = semantic_scholar_citations("ARXIV:1706.03762", start=-1)

        assert "start must be greater than or equal to 0" in result
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._semantic_scholar.urllib.request.urlopen")
    def test_no_citations(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_EMPTY_RESPONSE)

        result = semantic_scholar_citations("ARXIV:1706.03762")

        assert "No Semantic Scholar citations" in result
