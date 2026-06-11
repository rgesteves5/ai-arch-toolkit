"""Tests for toolkit/tools/_pubmed.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._pubmed import pubmed_article, pubmed_search

_ESEARCH_RESULT = {"esearchresult": {"idlist": ["26017442"]}}
_EMPTY_SEARCH_RESULT = {"esearchresult": {"idlist": []}}
_ARTICLE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>26017442</PMID>
      <Article>
        <Journal>
          <Title>Nature</Title>
          <JournalIssue>
            <PubDate>
              <Year>2015</Year>
              <Month>May</Month>
              <Day>28</Day>
            </PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Deep <i>learning</i></ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">
            Deep learning allows computational models.
          </AbstractText>
          <AbstractText>It is useful in many domains.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <ForeName>Yann</ForeName>
            <LastName>LeCun</LastName>
            <Initials>Y</Initials>
          </Author>
          <Author>
            <ForeName>Yoshua</ForeName>
            <LastName>Bengio</LastName>
          </Author>
        </AuthorList>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
          <PublicationType>Review</PublicationType>
        </PublicationTypeList>
      </Article>
      <MeshHeadingList>
        <MeshHeading>
          <DescriptorName>Machine Learning</DescriptorName>
          <QualifierName>methods</QualifierName>
        </MeshHeading>
        <MeshHeading>
          <DescriptorName>Neural Networks, Computer</DescriptorName>
        </MeshHeading>
      </MeshHeadingList>
      <KeywordList>
        <Keyword>deep learning</Keyword>
        <Keyword>neural networks</Keyword>
      </KeywordList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">26017442</ArticleId>
        <ArticleId IdType="doi">10.1038/nature14539</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

_EMPTY_ARTICLE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet/>
"""


def _mock_urlopen(content: str | dict):
    resp = MagicMock()
    if isinstance(content, dict):
        resp.read.return_value = json.dumps(content).encode()
    else:
        resp.read.return_value = content.encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _called_request(mock_urlopen, index: int = 0):
    return mock_urlopen.call_args_list[index].args[0]


def _called_params(mock_urlopen, index: int = 0) -> dict[str, list[str]]:
    return parse_qs(urlparse(_called_request(mock_urlopen, index).full_url).query)


class TestPubmedSearch:
    @patch("ai_arch_toolkit.toolkit.tools._pubmed.urllib.request.urlopen")
    def test_returns_results(self, mock_urlopen):
        mock_urlopen.side_effect = [_mock_urlopen(_ESEARCH_RESULT), _mock_urlopen(_ARTICLE_XML)]

        result = pubmed_search("deep learning", max_results=2)

        assert "PubMed results for 'deep learning'" in result
        assert "Deep learning" in result
        assert "PMID: 26017442 | DOI: 10.1038/nature14539 | published: 2015-05-28" in result
        assert "Yann LeCun, Yoshua Bengio" in result
        assert "Journal: Nature" in result
        assert "https://pubmed.ncbi.nlm.nih.gov/26017442/" in result
        assert "Abstract:" not in result

        search_params = _called_params(mock_urlopen, 0)
        assert search_params["db"] == ["pubmed"]
        assert search_params["term"] == ["deep learning"]
        assert search_params["retmode"] == ["json"]
        assert search_params["retstart"] == ["0"]
        assert search_params["retmax"] == ["2"]
        assert search_params["sort"] == ["relevance"]
        assert search_params["tool"] == ["ai_arch_toolkit"]

        fetch_params = _called_params(mock_urlopen, 1)
        assert fetch_params["id"] == ["26017442"]
        assert fetch_params["retmode"] == ["xml"]

    @patch("ai_arch_toolkit.toolkit.tools._pubmed.urllib.request.urlopen")
    def test_filters_dates_sort_start_and_caps_max_results(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_EMPTY_SEARCH_RESULT)

        pubmed_search(
            '"large language models"[Title/Abstract]',
            max_results=99,
            start=40,
            from_date="2024-01-01",
            to_date="2024-12-31",
            sort="pub_date",
        )

        params = _called_params(mock_urlopen)
        assert params["term"] == ['"large language models"[Title/Abstract]']
        assert params["retmax"] == ["20"]
        assert params["retstart"] == ["40"]
        assert params["sort"] == ["pub date"]
        assert params["datetype"] == ["pdat"]
        assert params["mindate"] == ["2024/01/01"]
        assert params["maxdate"] == ["2024/12/31"]
        assert mock_urlopen.call_count == 1

    @patch("ai_arch_toolkit.toolkit.tools._pubmed.urllib.request.urlopen")
    def test_no_results(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_EMPTY_SEARCH_RESULT)

        result = pubmed_search("no such paper")

        assert "No PubMed results" in result

    @patch("ai_arch_toolkit.toolkit.tools._pubmed.urllib.request.urlopen")
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "query cannot be empty" in pubmed_search("")
        assert "start must be greater than or equal to 0" in pubmed_search("test", start=-1)
        assert "invalid from_date" in pubmed_search("test", from_date="01-01-2024")
        assert "from_date must be before" in pubmed_search(
            "test", from_date="2024-02-01", to_date="2024-01-01"
        )
        assert "sort must be one of" in pubmed_search("test", sort="best")
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._pubmed.urllib.request.urlopen")
    def test_api_failure(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()

        result = pubmed_search("test")

        assert "timed out" in result.lower()

    @patch("ai_arch_toolkit.toolkit.tools._pubmed.urllib.request.urlopen")
    def test_search_parse_failure(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen("not json")

        result = pubmed_search("test")

        assert "could not parse API response" in result

    @patch("ai_arch_toolkit.toolkit.tools._pubmed.urllib.request.urlopen")
    def test_article_xml_parse_failure(self, mock_urlopen):
        mock_urlopen.side_effect = [_mock_urlopen(_ESEARCH_RESULT), _mock_urlopen("<not xml")]

        result = pubmed_search("test")

        assert "could not parse article XML" in result


class TestPubmedArticle:
    @patch("ai_arch_toolkit.toolkit.tools._pubmed.urllib.request.urlopen")
    def test_returns_article_by_pmid(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_ARTICLE_XML)

        result = pubmed_article("26017442")

        assert result.startswith("PubMed article 26017442:")
        assert "Deep learning" in result
        assert "Abstract: BACKGROUND: Deep learning allows computational models." in result
        assert "It is useful in many domains." in result
        assert "Publication types: Journal Article, Review" in result
        assert "MeSH: Machine Learning (methods), Neural Networks, Computer" in result
        assert "Keywords: deep learning, neural networks" in result
        assert "1. Deep learning" not in result

        params = _called_params(mock_urlopen)
        assert params["id"] == ["26017442"]

    @patch("ai_arch_toolkit.toolkit.tools._pubmed.urllib.request.urlopen")
    def test_invalid_pmid(self, mock_urlopen):
        result = pubmed_article("PMID 26017442")

        assert "invalid PMID" in result
        mock_urlopen.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._pubmed.urllib.request.urlopen")
    def test_not_found(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(_EMPTY_ARTICLE_XML)

        result = pubmed_article("999999999")

        assert "not found" in result.lower()
