"""Tests for toolkit/tools/_arxiv.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._arxiv import arxiv_paper, arxiv_search

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


def _mock_urlopen(content: str):
    resp = MagicMock()
    resp.read.return_value = content.encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _called_params(mock_urlopen) -> dict[str, list[str]]:
    request = mock_urlopen.call_args.args[0]
    return parse_qs(urlparse(request.full_url).query)


class TestArxivSearch:
    @patch("ai_arch_toolkit.toolkit.tools._arxiv.urllib.request.urlopen")
    def test_returns_results(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_ATOM_FEED)

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

    @patch("ai_arch_toolkit.toolkit.tools._arxiv.urllib.request.urlopen")
    def test_filters_category_dates_start_and_caps_max_results(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_EMPTY_FEED)

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

    @patch("ai_arch_toolkit.toolkit.tools._arxiv.urllib.request.urlopen")
    def test_keeps_advanced_query_syntax(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_EMPTY_FEED)

        arxiv_search('ti:"language agents" AND cat:cs.AI')

        params = _called_params(mock_urlopen)
        assert params["search_query"] == ['(ti:"language agents" AND cat:cs.AI)']

    @patch("ai_arch_toolkit.toolkit.tools._arxiv.urllib.request.urlopen")
    def test_no_results(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_EMPTY_FEED)

        result = arxiv_search("no such paper")

        assert "No arXiv results" in result

    @patch("ai_arch_toolkit.toolkit.tools._arxiv.urllib.request.urlopen")
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        result = arxiv_search("test", category="bad category")

        assert "invalid category" in result
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._arxiv.urllib.request.urlopen")
    def test_invalid_date(self, mock_urlopen):
        result = arxiv_search("test", from_date="01-01-2024")

        assert "invalid from_date" in result
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._arxiv.urllib.request.urlopen")
    def test_rejects_negative_start(self, mock_urlopen):
        result = arxiv_search("test", start=-1)

        assert "start must be greater than or equal to 0" in result
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._arxiv.urllib.request.urlopen")
    def test_rejects_reversed_date_range(self, mock_urlopen):
        result = arxiv_search("test", from_date="2024-02-01", to_date="2024-01-01")

        assert "from_date must be before or equal to to_date" in result
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._arxiv.urllib.request.urlopen")
    def test_api_failure(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()

        result = arxiv_search("test")

        assert "timed out" in result.lower()

    @patch("ai_arch_toolkit.toolkit.tools._arxiv.urllib.request.urlopen")
    def test_parse_failure(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen("<not xml")

        result = arxiv_search("test")

        assert "could not parse" in result


class TestArxivPaper:
    @patch("ai_arch_toolkit.toolkit.tools._arxiv.urllib.request.urlopen")
    def test_returns_paper_by_url(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_ATOM_FEED)

        result = arxiv_paper("https://arxiv.org/abs/1706.03762v7")

        assert result.startswith("arXiv paper 1706.03762v7:")
        assert "Attention Is All You Need" in result
        assert "1. Attention" not in result

        params = _called_params(mock_urlopen)
        assert params["id_list"] == ["1706.03762v7"]
        assert params["max_results"] == ["1"]

    @patch("ai_arch_toolkit.toolkit.tools._arxiv.urllib.request.urlopen")
    def test_normalizes_pdf_url(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_ATOM_FEED)

        arxiv_paper("https://arxiv.org/pdf/1706.03762v7.pdf")

        params = _called_params(mock_urlopen)
        assert params["id_list"] == ["1706.03762v7"]

    @patch("ai_arch_toolkit.toolkit.tools._arxiv.urllib.request.urlopen")
    def test_invalid_id(self, mock_urlopen):
        result = arxiv_paper("bad id")

        assert "invalid arXiv ID" in result
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._arxiv.urllib.request.urlopen")
    def test_not_found(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_EMPTY_FEED)

        result = arxiv_paper("2501.00000")

        assert "not found" in result
