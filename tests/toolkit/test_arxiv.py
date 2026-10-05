"""Tests for toolkit/tools/_arxiv.py."""

from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._arxiv import arxiv_paper, arxiv_search
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

_ATOM_FEED = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <updated>2023-08-02T00:00:00Z</updated>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All You Need</title>
    <summary>
      The dominant sequence transduction models are based on complex recurrent
      or convolutional neural networks.
    </summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <arxiv:comment>15 pages, 5 figures</arxiv:comment>
    <arxiv:journal_ref>NeurIPS 2017</arxiv:journal_ref>
    <arxiv:doi>10.48550/arXiv.1706.03762</arxiv:doi>
    <link href="http://arxiv.org/abs/1706.03762v7" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/1706.03762v7" rel="related"
      type="application/pdf"/>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.CL"/>
    <category term="cs.LG"/>
  </entry>
</feed>
"""

_EMPTY_FEED = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>Empty arXiv query</title>
</feed>
"""


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    request = mock_urlopen.call_args.args[0]
    return parse_qs(urlparse(request.full_url).query)


class TestArxivSearch:
    @patch(HTTP_OPEN)
    def test_returns_results(self, mock_urlopen):
        mock_urlopen.return_value = respond(_ATOM_FEED)

        result = arxiv_search("transformers", max_results=2)

        assert "arXiv results for 'transformers'" in result
        assert "Attention Is All You Need" in result
        assert "Ashish Vaswani, Noam Shazeer" in result
        assert "arXiv: 1706.03762v7 | cs.CL" in result
        assert "published: 2017-06-12" in result
        assert "updated: 2023-08-02" in result
        assert "DOI: 10.48550/arXiv.1706.03762" in result
        assert "https://arxiv.org/abs/1706.03762v7" in result
        assert "https://arxiv.org/pdf/1706.03762v7" in result

        params = _called_params(mock_urlopen)
        assert params["search_query"] == ['all:"transformers"']
        assert params["start"] == ["0"]
        assert params["max_results"] == ["2"]
        assert params["sortBy"] == ["relevance"]
        assert params["sortOrder"] == ["descending"]

    @patch(HTTP_OPEN)
    def test_filters_category_dates_start_and_caps_max_results(self, mock_urlopen):
        mock_urlopen.return_value = respond(_EMPTY_FEED)

        arxiv_search(
            "language agents",
            max_results=99,
            start=40,
            category="cs.AI",
            sort_by="submittedDate",
            from_date="2024-01-01",
            to_date="2024-01-31",
        )

        params = _called_params(mock_urlopen)
        assert params["max_results"] == ["20"]
        assert params["start"] == ["40"]
        assert params["sortBy"] == ["submittedDate"]
        assert params["search_query"] == [
            'cat:cs.AI AND all:"language agents" AND submittedDate:[202401010000 TO 202401312359]'
        ]

    @patch(HTTP_OPEN)
    def test_keeps_advanced_query_syntax(self, mock_urlopen):
        mock_urlopen.return_value = respond(_EMPTY_FEED)

        arxiv_search('ti:"language agents" AND cat:cs.AI')

        params = _called_params(mock_urlopen)
        assert params["search_query"] == ['(ti:"language agents" AND cat:cs.AI)']

    @patch(HTTP_OPEN)
    def test_no_results(self, mock_urlopen):
        mock_urlopen.return_value = respond(_EMPTY_FEED)

        result = arxiv_search("no such paper")

        assert "No arXiv results" in result

    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        with pytest.raises(ToolFailure) as caught:
            arxiv_search("test", category="bad category")

        assert caught.value.error.type == "validation_error"
        assert "invalid category" in caught.value.error.message
        mock_urlopen.assert_not_called()

    @pytest.mark.parametrize(
        ("kwargs", "words"),
        [
            ({"query": "  "}, "query cannot be empty"),
            ({"query": "x", "sort_by": "date"}, "invalid sort_by 'date'"),
            ({"query": "x", "sort_order": "up"}, "invalid sort_order 'up'"),
        ],
    )
    @patch(HTTP_OPEN)
    def test_other_invalid_arguments(self, mock_urlopen, kwargs, words):
        with pytest.raises(ToolFailure) as caught:
            arxiv_search(**kwargs)

        assert caught.value.error.type == "validation_error"
        assert words in caught.value.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_invalid_date(self, mock_urlopen):
        with pytest.raises(ToolFailure) as caught:
            arxiv_search("test", from_date="01-01-2024")

        assert caught.value.error.type == "validation_error"
        assert "invalid from_date" in caught.value.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_rejects_negative_start(self, mock_urlopen):
        with pytest.raises(ToolFailure) as caught:
            arxiv_search("test", start=-1)

        assert caught.value.error.type == "validation_error"
        assert "start must be greater than or equal to 0" in caught.value.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_rejects_reversed_date_range(self, mock_urlopen):
        with pytest.raises(ToolFailure) as caught:
            arxiv_search("test", from_date="2024-02-01", to_date="2024-01-01")

        assert caught.value.error.type == "validation_error"
        assert "must be before or equal to to_date" in caught.value.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_api_failure(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()

        with pytest.raises(ToolFailure) as caught:
            arxiv_search("test")

        assert caught.value.error.type == "upstream"
        assert caught.value.error.retryable
        assert "timed out" in caught.value.error.message.lower()

    @patch(HTTP_OPEN)
    def test_parse_failure(self, mock_urlopen):
        mock_urlopen.return_value = respond("<not xml")

        with pytest.raises(ToolFailure) as caught:
            arxiv_search("test")

        assert caught.value.error.type == "upstream"
        assert "could not parse" in caught.value.error.message


class TestArxivPaper:
    @patch(HTTP_OPEN)
    def test_returns_paper_by_url(self, mock_urlopen):
        mock_urlopen.return_value = respond(_ATOM_FEED)

        result = arxiv_paper("https://arxiv.org/abs/1706.03762v7")

        assert result.startswith("arXiv paper 1706.03762v7:")
        assert "Attention Is All You Need" in result
        assert "1. Attention" not in result

        params = _called_params(mock_urlopen)
        assert params["id_list"] == ["1706.03762v7"]
        assert params["max_results"] == ["1"]

    @patch(HTTP_OPEN)
    def test_normalizes_pdf_url(self, mock_urlopen):
        mock_urlopen.return_value = respond(_ATOM_FEED)

        arxiv_paper("https://arxiv.org/pdf/1706.03762v7.pdf")

        params = _called_params(mock_urlopen)
        assert params["id_list"] == ["1706.03762v7"]

    @patch(HTTP_OPEN)
    def test_invalid_id(self, mock_urlopen):
        with pytest.raises(ToolFailure) as caught:
            arxiv_paper("bad id")

        assert caught.value.error.type == "validation_error"
        assert "invalid arXiv ID" in caught.value.error.message
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_not_found(self, mock_urlopen):
        mock_urlopen.return_value = respond(_EMPTY_FEED)

        with pytest.raises(ToolFailure) as caught:
            arxiv_paper("2501.00000")

        assert caught.value.error.type == "not_found"
        assert "no arXiv paper with ID 2501.00000" in caught.value.error.message
        assert "arxiv_search" in caught.value.error.message


# What the API answered live, with HTTP 400, to a query it could not read (2026-09-30).
_ERROR_FEED = b"""\
<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" \
xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns="http://www.w3.org/2005/Atom">
  <id>https://arxiv.org/</id>
  <title>arXiv Search Results</title>
  <updated>2026-09-29T23:41:08Z</updated>
  <opensearch:itemsPerPage>1</opensearch:itemsPerPage>
  <opensearch:totalResults>1</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <entry>
    <id>https://arxiv.org/api/errors</id>
    <title>Error</title>
    <updated>2026-09-29T23:41:08Z</updated>
    <link href="https://arxiv.org/api/errors" rel="alternate" type="text/html"/>
    <summary>Invalid query string: '( ( )'</summary>
    <author>
      <name>arXiv api core</name>
    </author>
  </entry>
</feed>
"""


@patch(HTTP_OPEN)
def test_a_query_arxiv_cannot_read_is_explained(mock_urlopen):
    # It read as "HTTP error 400: Bad Request".
    mock_urlopen.side_effect = http_error(400, "Bad Request", body=_ERROR_FEED)

    with pytest.raises(ToolFailure) as caught:
        arxiv_search("ti:(")

    assert caught.value.error.type == "validation_error"
    assert not caught.value.error.retryable
    assert caught.value.error.details == {"status": 400}
    assert caught.value.error.message.startswith(
        "arXiv rejected the request: Invalid query string: '( ( )'; check"
    )


@patch(HTTP_OPEN)
def test_an_error_entry_is_the_error_not_a_paper(mock_urlopen):
    # The form the user manual documents, with HTTP 200, would have read as a paper titled
    # "Error".
    mock_urlopen.return_value = respond(
        _ERROR_FEED.decode()
        .replace("https://arxiv.org/api/errors<", "http://arxiv.org/api/errors#bad_id<")
        .replace("Invalid query string: '( ( )'", "incorrect id format for 1234.1234")
    )

    with pytest.raises(ToolFailure) as caught:
        arxiv_paper("1234.1234")

    assert caught.value.error.type == "validation_error"
    assert caught.value.error.message.startswith(
        "arXiv rejected the request: incorrect id format for 1234.1234; check"
    )


@patch(HTTP_OPEN)
def test_an_error_feed_with_a_server_status_is_retryable_upstream(mock_urlopen):
    mock_urlopen.side_effect = http_error(503, "Service Unavailable", body=_ERROR_FEED)

    with pytest.raises(ToolFailure) as caught:
        arxiv_search("agents")

    assert caught.value.error.type == "upstream"
    assert caught.value.error.retryable
    assert caught.value.error.message == "HTTP error 503: Invalid query string: '( ( )'"


@patch(HTTP_OPEN)
def test_a_404_is_endpoint_not_found(mock_urlopen):
    # The query endpoint answers an unknown ID with an empty feed, so a 404 means it moved.
    mock_urlopen.side_effect = http_error(404, "Not Found")

    with pytest.raises(ToolFailure) as caught:
        arxiv_paper("1706.03762")

    assert caught.value.error.type == "upstream"
    assert "arXiv: endpoint not found (HTTP 404)" in caught.value.error.message
