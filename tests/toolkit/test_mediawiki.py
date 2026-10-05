"""Tests for toolkit/tools/_mediawiki.py: the error reader and the wikis the tools may read."""

from __future__ import annotations

import importlib
import pkgutil

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit import tools
from ai_arch_toolkit.toolkit.tools._http import Api, Reply
from ai_arch_toolkit.toolkit.tools._mediawiki import (
    WIKIPEDIA,
    WIKTIONARY,
    mediawiki_error,
    wiki_api,
    wiki_host,
)
from tests.toolkit.wiki_pages import api_error


class TestTheErrorReader:
    """MediaWiki answers errors with an ``error`` object, usually with HTTP 200."""

    @pytest.mark.parametrize(
        ("data", "error"),
        [
            (api_error("readonly", "Read-only mode."), "readonly: Read-only mode."),
            ({"error": {"code": "readonly"}}, "readonly"),
            ({"error": {"info": "Read-only mode."}}, "Read-only mode."),
            ({"error": {"code": None, "info": ""}}, "unknown error"),
            ({"parse": {"title": "apple"}}, None),
            ({"error": "not an object"}, None),
            ([{"error": {"code": "x"}}], None),
            (None, None),
        ],
    )
    def test_it_reads_the_error_object(self, data: object, error: str | None) -> None:
        assert mediawiki_error(Reply(status=200, headers={}, body=data)) == error

    @pytest.mark.parametrize(
        ("code", "kind", "retryable", "message"),
        [
            (
                "missingtitle",
                "not_found",
                False,
                "no such page (missingtitle: Some words); "
                "find the exact title with the wiki's search",
            ),
            (
                "invalidtitle",
                "validation_error",
                False,
                "not a valid page title (invalidtitle: Some words); "
                "give a title as the wiki's search returns it",
            ),
            (
                "pagecannotexist",
                "validation_error",
                False,
                "not a page the wiki can hold (pagecannotexist: Some words); give an article's "
                "title, not a special page",
            ),
            (
                "ratelimited",
                "rate_limited",
                True,
                "the wiki asked to slow down (ratelimited: Some words); try again later",
            ),
            (
                "maxlag",
                "rate_limited",
                True,
                "the wiki asked to slow down (maxlag: Some words); try again later",
            ),
        ],
    )
    def test_it_types_the_codes_that_say_what_happened(
        self, code: str, kind: str, retryable: bool, message: str
    ) -> None:
        error = mediawiki_error(Reply(status=200, headers={}, body=api_error(code, "Some words.")))

        assert isinstance(error, ToolFailure)
        assert (error.error.type, error.error.retryable) == (kind, retryable)
        assert error.error.message == message

    def test_it_reads_the_code_in_the_header_without_an_error_object(self) -> None:
        headers = {"mediawiki-api-error": "ratelimited"}

        error = mediawiki_error(Reply(status=503, headers=headers, body="<html>busy</html>"))
        plain = mediawiki_error(Reply(status=503, headers={"x": "y"}, body="<html>busy</html>"))

        assert isinstance(error, ToolFailure)
        assert error.error.type == "rate_limited"
        assert plain is None


class TestTheWikis:
    @pytest.mark.parametrize(
        ("wiki", "host"),
        [
            ("en.wikipedia.org", "en.wikipedia.org"),
            (" PT.Wikipedia.org ", "pt.wikipedia.org"),
            ("https://en.wikibooks.org", "en.wikibooks.org"),
            ("https://en.wikibooks.org/w/api.php", "en.wikibooks.org"),
            ("www.wikidata.org", "www.wikidata.org"),
            ("mediawiki.org", "mediawiki.org"),
        ],
    )
    def test_a_wikimedia_wiki_is_named_by_its_host_or_url(self, wiki: str, host: str) -> None:
        assert wiki_host(wiki) == host
        assert wiki_api(wiki).base == f"https://{host}/w/api.php"

    @pytest.mark.parametrize(
        "wiki",
        [
            "",
            "evil.example",
            "en.wikipedia.org.evil.example",
            "evilwikipedia.org",
            "169.254.169.254",
            "localhost",
            "user:pw@en.wikipedia.org",
            "en.wikipedia.org:443",
            "wikipedia",
            "en wikipedia org",
        ],
    )
    def test_any_other_host_is_refused(self, wiki: str) -> None:
        with pytest.raises(ToolFailure) as caught:
            wiki_api(wiki)

        assert caught.value.error.type == "validation_error"
        assert "invalid wiki" in caught.value.error.message

    def test_every_wiki_takes_the_wikis_own_answer_size_and_pace(self) -> None:
        for api in (wiki_api(WIKIPEDIA), wiki_api("en.wikibooks.org")):
            assert api.max_bytes == 13 * 2**20
            assert api.min_interval_s == 0.25

    def test_the_tools_own_wikis_take_a_404_for_an_endpoint_that_moved(self) -> None:
        assert not wiki_api(WIKIPEDIA).caller_base
        assert not wiki_api(WIKTIONARY).caller_base
        assert wiki_api("en.wikibooks.org").caller_base  # a host the caller chose


def test_every_api_on_a_mediawiki_endpoint_reads_its_error_object() -> None:
    # A new MediaWiki client that forgets the reader would take an error for an empty result.
    apis = {}
    for info in pkgutil.iter_modules(tools.__path__):
        module = importlib.import_module(f"{tools.__name__}.{info.name}")
        apis.update(
            (f"{info.name}.{name}", value)
            for name, value in vars(module).items()
            if isinstance(value, Api) and value.base.endswith("api.php")
        )
    apis["wiki_api(en.wikibooks.org)"] = wiki_api("en.wikibooks.org")

    assert set(apis) >= {
        "_geo._WIKIDATA",
        "_mediawiki._WIKIPEDIA_API",
        "_mediawiki._WIKTIONARY_API",
        "_wikidata._API",
    }
    assert {name for name, api in apis.items() if api.error_reader is not mediawiki_error} == set()
