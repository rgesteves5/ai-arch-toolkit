"""Tests for toolkit/tools/_mediawiki.py."""

from __future__ import annotations

import importlib
import pkgutil
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit import tools
from ai_arch_toolkit.toolkit.tools._http import Api
from ai_arch_toolkit.toolkit.tools._mediawiki import (
    mediawiki_error,
    mediawiki_page,
    mediawiki_search,
    mediawiki_sections,
    wiktionary_entry,
)
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

_WIKIBOOKS = "https://en.wikibooks.org/w/api.php"
_WIKTIONARY = "https://en.wiktionary.org/w/api.php"
_BOOK_PAGE = {"title": "Creative Writing/Novels/Editing", "api_url": _WIKIBOOKS}


def _failure(fn, *args, **kwargs) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        fn(*args, **kwargs)
    return caught.value


def _params(mock_urlopen):
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


def _api_error(code: str, info: str) -> dict[str, object]:
    """An error answer as MediaWiki sends it, with HTTP 200 (en.wikibooks.org, 2026-09-29)."""
    return {
        "error": {
            "code": code,
            "info": info,
            "*": "See https://en.wikibooks.org/w/api.php for API usage.",
        },
        "warnings": {"parse": {"*": '"prop=sections" has been deprecated.'}},
        "servedby": "mw-api-ext.eqiad.main-79f4dd7c47-hgftf",
    }


_MISSING_TITLE = _api_error("missingtitle", "The page you specified doesn't exist.")


class TestMediaWiki:
    @patch(HTTP_OPEN)
    def test_search(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            {
                "query": {
                    "searchinfo": {"totalhits": 1},
                    "search": [{"title": "apple", "pageid": 1, "snippet": "A <b>fruit</b>"}],
                }
            }
        )

        result = mediawiki_search("apple")

        assert "apple | pageid: 1" in result
        assert "A fruit" in result
        assert _params(mock_urlopen)["action"] == ["query"]

    @patch(HTTP_OPEN)
    def test_page_sections_and_wiktionary(self, mock_urlopen):
        payload = {
            "parse": {
                "title": "apple",
                "wikitext": {"*": "==English==\n===Noun===\n# A [[fruit]]."},
                "sections": [{"index": "1", "line": "English", "level": "2"}],
            }
        }
        mock_urlopen.return_value = respond(payload)
        assert "A fruit." in mediawiki_page("apple")

        mock_urlopen.return_value = respond(payload)
        assert "1. English | level: 2" in mediawiki_sections("apple")

        mock_urlopen.return_value = respond(payload)
        result = wiktionary_entry("apple")
        assert "Wiktionary entry apple (English):" in result
        assert "Noun:" in result

    @patch(HTTP_OPEN)
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        bad_url = _failure(mediawiki_search, "x", api_url="http://example.com/api.php")
        bad_term = _failure(wiktionary_entry, "bad<>")
        bad_language = _failure(wiktionary_entry, "apple", language="Fran<ais")
        bad_offset = _failure(mediawiki_search, "apple", offset=-1)
        not_api_php = _failure(mediawiki_page, "apple", api_url="https://en.wikipedia.org/w/")

        assert bad_url.error.type == "validation_error"
        assert "invalid api_url" in bad_url.error.message
        assert "https://en.wikipedia.org/w/api.php" in bad_url.error.message
        assert bad_term.error.type == "validation_error"
        assert "invalid term" in bad_term.error.message
        assert bad_language.error.type == "validation_error"
        assert "invalid language" in bad_language.error.message
        assert bad_offset.error.type == "validation_error"
        assert "offset" in bad_offset.error.message
        assert not_api_php.error.type == "validation_error"
        mock_urlopen.assert_not_called()


class TestApiErrors:
    """MediaWiki answers errors with HTTP 200 and an ``error`` object instead of the result."""

    @pytest.mark.parametrize(
        ("fn", "args", "endpoint", "message"),
        [
            (
                mediawiki_page,
                _BOOK_PAGE,
                _WIKIBOOKS,
                "en.wikibooks.org has no page titled 'Creative Writing/Novels/Editing'",
            ),
            (
                mediawiki_sections,
                _BOOK_PAGE,
                _WIKIBOOKS,
                "en.wikibooks.org has no page titled 'Creative Writing/Novels/Editing'",
            ),
            (
                wiktionary_entry,
                {"term": "zzqqxxnotaword"},
                _WIKTIONARY,
                "en.wiktionary.org has no page titled 'zzqqxxnotaword'",
            ),
        ],
    )
    @patch(HTTP_OPEN)
    def test_a_missing_page_is_not_found(self, mock_urlopen, fn, args, endpoint, message):
        mock_urlopen.return_value = respond(_MISSING_TITLE)

        failure = _failure(fn, **args)

        assert failure.error.type == "not_found"
        assert not failure.error.retryable
        assert failure.error.message == f"{message}; find the title with mediawiki_search"
        assert mock_urlopen.call_args.args[0].full_url.startswith(f"{endpoint}?")

    @patch(HTTP_OPEN)
    def test_another_error_the_api_reports_is_upstream(self, mock_urlopen):
        mock_urlopen.return_value = respond(_api_error("readonly", "Read-only mode."))

        failure = _failure(mediawiki_page, "apple")

        assert failure.error.type == "upstream"
        assert str(failure) == "readonly: Read-only mode."

    @patch(HTTP_OPEN)
    def test_a_search_the_api_refuses_fails_instead_of_finding_nothing(self, mock_urlopen):
        mock_urlopen.return_value = respond(
            _api_error(
                "cirrussearch-offset-too-large",
                "Could not retrieve results. Up to 10000 search results are supported, but "
                "results starting at 99999 were requested.",
            )
        )

        failure = _failure(mediawiki_search, "apple", offset=99999)

        assert failure.error.type == "upstream"
        assert str(failure).startswith(
            "cirrussearch-offset-too-large: Could not retrieve results."
        )

    @patch(HTTP_OPEN)
    def test_a_search_with_no_hits_is_an_answer(self, mock_urlopen):
        mock_urlopen.return_value = respond({"query": {"search": []}})

        assert mediawiki_search("zzqqxx") == "No MediaWiki pages found."

    @pytest.mark.parametrize("fn", [mediawiki_page, mediawiki_sections, wiktionary_entry])
    @pytest.mark.parametrize("body", [{"batchcomplete": ""}, {"parse": None}, {"parse": []}])
    @patch(HTTP_OPEN)
    def test_a_parse_answer_without_a_parse_object_fails(self, mock_urlopen, body, fn):
        mock_urlopen.return_value = respond(body)

        failure = _failure(fn, "apple")

        assert failure.error.type == "upstream"
        assert str(failure) == 'could not parse API response: no "parse" object'

    @patch(HTTP_OPEN)
    def test_a_page_without_sections_still_says_so(self, mock_urlopen):
        mock_urlopen.return_value = respond({"parse": {"title": "Stub", "sections": []}})

        assert mediawiki_sections("Stub") == "No MediaWiki sections found for Stub."

    @patch(HTTP_OPEN)
    def test_a_status_error_keeps_its_own_message(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(404, "Not Found")

        failure = _failure(mediawiki_page, "apple")

        assert failure.error.type == "upstream"
        assert str(failure) == "no matching records found."

    @patch(HTTP_OPEN)
    def test_a_rate_limit_is_rate_limited(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(429, "Too Many Requests")

        failure = _failure(mediawiki_search, "apple")

        assert failure.error.type == "rate_limited"
        assert failure.error.retryable

    @pytest.mark.parametrize(
        ("data", "error"),
        [
            (_MISSING_TITLE, "missingtitle: The page you specified doesn't exist."),
            ({"error": {"code": "readonly"}}, "readonly"),
            ({"error": {"info": "Read-only mode."}}, "Read-only mode."),
            ({"error": {"code": None, "info": ""}}, "unknown error"),
            ({"parse": {"title": "apple"}}, None),
            ({"error": "not an object"}, None),
            ([{"error": {"code": "x"}}], None),
            (None, None),
        ],
    )
    def test_mediawiki_error_reads_the_error_object(self, data, error):
        assert mediawiki_error(data) == error


@pytest.mark.parametrize("fn", [mediawiki_search, mediawiki_page, mediawiki_sections])
@pytest.mark.parametrize(
    "url",
    [
        "https://169.254.169.254/api.php",
        "https://localhost:8443/x/api.php",
        "https://user:pw@en.wikipedia.org/w/api.php",
        "https://evil.example/api.php",
        "https://en.wikipedia.org:443/w/api.php",
        "https://en.wikipedia.org.evil.example/api.php",
        "https://evilwikipedia.org/api.php",
    ],
)
@patch(HTTP_OPEN)
def test_mediawiki_rejects_untrusted_hosts(mock_urlopen, fn, url):
    mock_urlopen.return_value = respond({})
    failure = _failure(fn, "apple", api_url=url)
    assert failure.error.type == "validation_error"
    assert "invalid api_url" in failure.error.message
    mock_urlopen.assert_not_called()


@patch(HTTP_OPEN)
def test_mediawiki_accepts_portuguese_wikipedia(mock_urlopen):
    mock_urlopen.return_value = respond({"query": {"search": []}})
    mediawiki_search("apple", api_url="https://pt.wikipedia.org/w/api.php")
    mock_urlopen.assert_called_once()


def test_every_api_on_a_mediawiki_endpoint_reads_its_error_object():
    # A new MediaWiki client that forgets the reader would take an error for an empty result.
    apis = {}
    for info in pkgutil.iter_modules(tools.__path__):
        module = importlib.import_module(f"{tools.__name__}.{info.name}")
        apis.update(
            (f"{info.name}.{name}", value)
            for name, value in vars(module).items()
            if isinstance(value, Api) and value.base.endswith("api.php")
        )

    assert set(apis) >= {
        "_geo._WIKIDATA",
        "_mediawiki._WIKTIONARY",
        "_wikidata._API",
        "_wikipedia._API",
    }
    assert {name for name, api in apis.items() if api.body_error is not mediawiki_error} == set()


@patch(HTTP_OPEN)
def test_an_invalid_title_is_a_validation_error(mock_urlopen):
    mock_urlopen.return_value = respond(_api_error("invalidtitle", 'Bad title "Talk:".'))

    failure = _failure(mediawiki_page, "Talk:")

    assert failure.error.type == "validation_error"
    assert "not a valid page title" in failure.error.message
