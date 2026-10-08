"""Tests for toolkit/tools/_semantic_scholar.py (T06)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._semantic_scholar import (
    semantic_scholar_citations,
    semantic_scholar_paper,
    semantic_scholar_search,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond
from tests.toolkit.literature_answers import (
    S2_UNACCEPTABLE,
    S2_UNRECOGNIZED,
    s2_citation,
    s2_citations,
    s2_paper,
    s2_search,
)

_ID = "649def34f8be52c8b66281af98ae884c09aef38b"


def _text(result: ToolResult) -> str:
    assert isinstance(result, ToolResult) and result.ok, result
    assert isinstance(result.value, str)
    return result.value


def _sent(mock_urlopen: MagicMock) -> tuple[str, dict[str, list[str]]]:
    url = urlparse(mock_urlopen.call_args.args[0].full_url)
    return url.path, parse_qs(url.query)


def _failure(fn: Any, *args: Any, **kwargs: Any) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


class TestSemanticScholarSearch:
    @patch(HTTP_OPEN)
    def test_a_page_says_the_total_and_the_next_start(self, mock_urlopen):
        mock_urlopen.return_value = respond(s2_search(s2_paper("a"), s2_paper("b"), total=4321))

        result = semantic_scholar_search("transformers", max_results=2)

        text = _text(result)
        assert text.splitlines()[:8] == [
            "Semantic Scholar papers that match 'transformers':",
            "1. Attention Is All You Need",
            "   paperId: a | year: 2017 | published: 2017-06-12",
            "   Authors: Ashish Vaswani, Noam Shazeer",
            "   Venue: Neural Information Processing Systems",
            "   Citations: 123456 | Influential citations: 25000 | References: 50",
            "   IDs: DOI:10.48550/arXiv.1706.03762 | ARXIV:1706.03762 | PMID:26017442 | "
            "CorpusId:13756489",
            "   Open PDF: https://arxiv.org/pdf/1706.03762",
        ]
        assert text.endswith("[results 1-2 of 4321 | next: start=2]")
        path, params = _sent(mock_urlopen)
        assert path.endswith("/paper/search")
        assert params["query"] == ["transformers"]
        assert params["limit"] == ["2"]
        assert params["offset"] == ["0"]

    @patch(HTTP_OPEN)
    def test_the_last_batch_has_no_next_and_says_end(self, mock_urlopen):
        mock_urlopen.return_value = respond(s2_search(s2_paper("c"), total=3, offset=2))

        text = _text(semantic_scholar_search("transformers", max_results=2, start=2))

        assert text.splitlines()[1] == "3. Attention Is All You Need"
        assert text.endswith("[results 3-3 of 3 | end]")

    @patch(HTTP_OPEN)
    def test_relevance_search_reaches_the_first_1000_results(self, mock_urlopen):
        # offset + limit is at most 1,000 since 2023-10-31 (https://github.com/allenai/s2-folks/
        # blob/main/API_RELEASE_NOTES.md): the last page asks for what is left.
        mock_urlopen.return_value = respond(
            s2_search(s2_paper("a"), s2_paper("b"), total=50_000, offset=997)
        )

        text = _text(semantic_scholar_search("transformers", max_results=5, start=997))

        assert _sent(mock_urlopen)[1]["limit"] == ["2"]
        assert text.endswith("[results 998-999 of 50000 | the rest cannot be read here]")

    @patch(HTTP_OPEN)
    def test_a_long_author_list_says_how_many_more_and_where(self, mock_urlopen):
        mock_urlopen.return_value = respond(s2_search(s2_paper(authors=11), total=1))

        text = _text(semantic_scholar_search("transformers"))

        assert f"(+3 more; semantic_scholar_paper('{_ID}') lists all)" in text

    @pytest.mark.parametrize(
        "query", ["10.1038/nature14539", "https://doi.org/10.1038/nature14539", "1706.03762"]
    )
    @patch(HTTP_OPEN)
    def test_an_identifier_is_looked_up_with_the_paper_tool_not_searched(
        self, mock_urlopen, query
    ):
        # The search takes plain text, "no special query syntax" (the API's documentation).
        failure = _failure(semantic_scholar_search, query)

        assert failure.error.type == "validation_error"
        assert f"semantic_scholar_paper({query!r})" in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_zero_results_say_so_with_the_query(self, mock_urlopen):
        mock_urlopen.return_value = respond(s2_search(total=0))

        assert _text(semantic_scholar_search("zzqq")) == (
            "No Semantic Scholar papers match 'zzqq'."
        )

    @patch(HTTP_OPEN)
    def test_filters_year_venue_and_start(self, mock_urlopen):
        mock_urlopen.return_value = respond(s2_search(total=0, offset=40))

        semantic_scholar_search("agents", max_results=20, start=40, year="2020-2024", venue="ACL")

        params = _sent(mock_urlopen)[1]
        assert params["limit"] == ["20"]
        assert params["offset"] == ["40"]
        assert params["year"] == ["2020-2024"]
        assert params["venue"] == ["ACL"]

    @pytest.mark.parametrize("year", ["twenty", "2020-21", "20200"])
    @patch(HTTP_OPEN)
    def test_a_year_the_api_does_not_take_is_refused_before_the_request(self, mock_urlopen, year):
        failure = _failure(semantic_scholar_search, "agents", year=year)

        assert failure.error.type == "validation_error"
        assert "invalid year" in failure.error.message
        mock_urlopen.assert_not_called()

    @pytest.mark.parametrize(
        ("max_results", "start", "kept"),
        [(20, 998, True), (21, 0, False), (0, 0, False), (5, 999, False)],
    )
    @patch(HTTP_OPEN)
    def test_the_limits_are_the_schemas(self, mock_urlopen, max_results, start, kept):
        mock_urlopen.return_value = respond(s2_search(total=0))
        call = ToolCall(
            id="c1",
            name="semantic_scholar_search",
            input={"query": "x", "max_results": max_results, "start": start},
        )

        assert ToolGroup(semantic_scholar_search).execute(call).ok is kept

    @patch(HTTP_OPEN)
    def test_an_unacceptable_parameter_is_the_callers_to_fix(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            400, "Bad Request", body=json.dumps(S2_UNACCEPTABLE).encode()
        )

        failure = _failure(semantic_scholar_search, "test")

        assert failure.error.type == "validation_error"
        assert failure.error.message == (
            "Semantic Scholar refused the request: Unacceptable query params: [year=twenty]; "
            "correct that parameter"
        )

    @patch(HTTP_OPEN)
    def test_another_refusal_keeps_the_sources_words(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            400, "Bad Request", body=json.dumps(S2_UNRECOGNIZED).encode()
        )

        failure = _failure(semantic_scholar_search, "test")

        assert failure.error.type == "upstream"
        assert (
            failure.error.message == "HTTP error 400: Unrecognized or unsupported fields: [nope]"
        )

    @patch(HTTP_OPEN)
    def test_a_429_without_a_key_says_how_to_get_one(self, mock_urlopen, monkeypatch):
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        mock_urlopen.side_effect = http_error(429, "Too Many Requests")

        failure = _failure(semantic_scholar_search, "test")

        assert failure.error.type == "rate_limited"
        assert failure.error.retryable
        assert "Set SEMANTIC_SCHOLAR_API_KEY" in failure.error.message
        assert "https://www.semanticscholar.org/product/api#api-key-form" in failure.error.message

    @patch(HTTP_OPEN)
    def test_a_key_in_the_environment_goes_in_x_api_key(self, mock_urlopen, monkeypatch):
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "s2-key")
        mock_urlopen.return_value = respond(s2_search(total=0))

        semantic_scholar_search("test")

        assert mock_urlopen.call_args.args[0].get_header("X-api-key") == "s2-key"

    @patch(HTTP_OPEN)
    def test_without_a_key_no_key_header_goes(self, mock_urlopen, monkeypatch):
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        mock_urlopen.return_value = respond(s2_search(total=0))

        semantic_scholar_search("test")

        assert mock_urlopen.call_args.args[0].get_header("X-api-key") is None

    @patch(HTTP_OPEN)
    def test_a_404_on_the_search_is_an_endpoint_that_moved_not_an_empty_result(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(semantic_scholar_search, "test")

        assert failure.error.type == "upstream"
        assert failure.error.message.startswith("Semantic Scholar: endpoint not found (HTTP 404)")

    @patch(HTTP_OPEN)
    def test_parse_failure(self, mock_urlopen):
        mock_urlopen.return_value = respond("not json")

        failure = _failure(semantic_scholar_search, "test")

        assert failure.error.type == "upstream"
        assert "could not parse" in failure.error.message


class TestSemanticScholarPaper:
    @patch(HTTP_OPEN)
    def test_the_record_is_whole(self, mock_urlopen):
        mock_urlopen.return_value = respond(s2_paper(authors=11))

        text = _text(semantic_scholar_paper("https://doi.org/10.48550/arXiv.1706.03762"))

        lines = text.splitlines()
        assert lines[:7] == [
            "Semantic Scholar paper DOI:10.48550/arXiv.1706.03762:",
            "Attention Is All You Need",
            f"paperId: {_ID} | year: 2017 | published: 2017-06-12",
            "Venue: Neural Information Processing Systems",
            "Citations: 123456 | Influential citations: 25000 | References: 50",
            "IDs: DOI:10.48550/arXiv.1706.03762 | ARXIV:1706.03762 | PMID:26017442 | "
            "CorpusId:13756489",
            "Publication types: Conference | Fields: Computer Science, Machine Learning",
        ]
        assert "Abstract: The dominant sequence transduction models" in text
        assert "Authors (11): Ashish Vaswani, Noam Shazeer, Author 3, " in text
        assert "Author 11" in text
        assert urlparse(mock_urlopen.call_args.args[0].full_url).path.endswith(
            "/DOI:10.48550%2FarXiv.1706.03762"
        )

    @patch(HTTP_OPEN)
    def test_a_long_abstract_reads_on_through_the_window(self, mock_urlopen):
        long = "A sentence of the abstract. " * 100
        mock_urlopen.side_effect = [respond(s2_paper(abstract=long)) for _ in range(2)]

        first = semantic_scholar_paper(_ID, max_chars=1000)
        last = first.metadata["window"]["last"]
        second = semantic_scholar_paper(_ID, max_chars=1000, offset=last)

        assert _text(first).endswith(f"next: offset={last}]")
        assert second.metadata["window"]["first"] == last

    @pytest.mark.parametrize(
        ("given", "path"),
        [
            ("https://arxiv.org/pdf/1706.03762v7.pdf", "/ARXIV:1706.03762v7"),
            ("26017442", "/PMID:26017442"),
            ("CorpusId:13756489", "/CorpusId:13756489"),
            ("doi:10.1/x", "/DOI:10.1%2Fx"),
            ("PMCID:PMC4567", "/PMCID:PMC4567"),
            (_ID, f"/{_ID}"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_the_identifier_forms_it_takes(self, mock_urlopen, given, path):
        mock_urlopen.return_value = respond(s2_paper())

        semantic_scholar_paper(given)

        assert urlparse(mock_urlopen.call_args.args[0].full_url).path.endswith(path)

    @pytest.mark.parametrize("given", ["", "PMID:abc", "CorpusId:x"])
    @patch(HTTP_OPEN)
    def test_invalid_paper_id(self, mock_urlopen, given):
        failure = _failure(semantic_scholar_paper, given)

        assert failure.error.type == "validation_error"
        assert "invalid paper_id" in failure.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            404, "Not Found", body=b'{"error": "Paper with id missing not found"}'
        )

        failure = _failure(semantic_scholar_paper, "missing")

        assert failure.error.type == "not_found"
        assert failure.error.message == (
            "no Semantic Scholar paper with ID missing; search with semantic_scholar_search."
        )

    @patch(HTTP_OPEN)
    def test_another_error_status_stays_the_sources_failure(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(500, "Internal Server Error")

        failure = _failure(semantic_scholar_paper, "x")

        assert failure.error.type == "upstream"
        assert failure.error.retryable


class TestSemanticScholarCitations:
    @patch(HTTP_OPEN)
    def test_a_page_says_there_is_more_and_the_next_start(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            s2_citations(s2_citation("c1", contexts=3), s2_citation("c2"), offset=10, more=True)
        )

        result = semantic_scholar_citations(_ID, max_results=2, start=10)

        text = _text(result)
        assert text.splitlines()[0] == f"Papers in Semantic Scholar that cite {_ID}:"
        assert text.splitlines()[1] == "11. A Paper That Cites It"
        assert "   Citation: influential | intents: background, methodology" in text
        for n in (1, 2, 3):  # every context, whole
            assert f'   Context: "Context {n} of the citation, following the Transformer."' in text
        assert text.endswith("[results 11-12 | next: start=12]")
        path, params = _sent(mock_urlopen)
        assert path.endswith(f"/paper/{_ID}/citations")
        assert params["offset"] == ["10"]
        assert params["limit"] == ["2"]

    @patch(HTTP_OPEN)
    def test_the_last_batch_says_end(self, mock_urlopen):
        mock_urlopen.return_value = respond(s2_citations(s2_citation(), offset=4))

        text = _text(semantic_scholar_citations(_ID, max_results=2, start=4))

        assert text.endswith("[results 5-5 | end]")

    @patch(HTTP_OPEN)
    def test_no_citations_say_so(self, mock_urlopen):
        mock_urlopen.return_value = respond(s2_citations())

        assert _text(semantic_scholar_citations(_ID)) == (
            f"No papers in Semantic Scholar cite {_ID}."
        )

    @pytest.mark.parametrize(
        ("max_results", "start", "kept"),
        [(20, 9_979, True), (21, 0, False), (5, 9_999, False), (5, -1, False)],
    )
    @patch(HTTP_OPEN)
    def test_the_limits_are_the_schemas(self, mock_urlopen, max_results, start, kept):
        mock_urlopen.return_value = respond(s2_citations())
        call = ToolCall(
            id="c1",
            name="semantic_scholar_citations",
            input={"paper_id": _ID, "max_results": max_results, "start": start},
        )

        assert ToolGroup(semantic_scholar_citations).execute(call).ok is kept

    @patch(HTTP_OPEN)
    def test_citations_of_an_unknown_paper_are_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(semantic_scholar_citations, "missing")

        assert failure.error.type == "not_found"
        assert "semantic_scholar_search" in failure.error.message


@patch(HTTP_OPEN)
def test_citations_past_the_apis_reach_say_the_rest_cannot_be_read_here(mock_urlopen):
    # Citations page through offset + limit < 10,000: past it, the paper's own count says how
    # many there are, instead of a footer that says "end" while Semantic Scholar has more.
    mock_urlopen.side_effect = [
        respond(s2_citations(s2_citation("a"), s2_citation("b"), offset=9_997, more=True)),
        respond({"paperId": _ID, "citationCount": 123456}),
    ]

    text = _text(semantic_scholar_citations(_ID, max_results=2, start=9_997))

    assert text.endswith("[results 9998-9999 of 123456 | the rest cannot be read here]")
    assert parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)["fields"] == [
        "citationCount"
    ]


@patch(HTTP_OPEN)
def test_a_paper_without_an_id_names_no_call_that_would_fail(mock_urlopen):
    paper = s2_paper(authors=11) | {"paperId": None}
    mock_urlopen.return_value = respond(s2_search(paper, total=1))

    text = _text(semantic_scholar_search("transformers"))

    assert "Author 8 (+3 more)" in text
    assert "semantic_scholar_paper(''" not in text
