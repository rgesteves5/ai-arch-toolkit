"""Tests for toolkit/tools/_mediawiki.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from ai_arch_toolkit.toolkit.tools._mediawiki import (
    mediawiki_page,
    mediawiki_search,
    mediawiki_sections,
    wiktionary_entry,
)


def _mock_urlopen(data: dict | str):
    resp = MagicMock()
    resp.read.return_value = (data if isinstance(data, str) else json.dumps(data)).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _params(mock_urlopen):
    return parse_qs(urlparse(mock_urlopen.call_args.args[0].full_url).query)


class TestMediaWiki:
    @patch("ai_arch_toolkit.toolkit.tools._mediawiki.urllib.request.urlopen")
    def test_search(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
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

    @patch("ai_arch_toolkit.toolkit.tools._mediawiki.urllib.request.urlopen")
    def test_page_sections_and_wiktionary(self, mock_urlopen):
        payload = {
            "parse": {
                "title": "apple",
                "wikitext": {"*": "==English==\n===Noun===\n# A [[fruit]]."},
                "sections": [{"index": "1", "line": "English", "level": "2"}],
            }
        }
        mock_urlopen.return_value = _mock_urlopen(payload)
        assert "A fruit." in mediawiki_page("apple")

        mock_urlopen.return_value = _mock_urlopen(payload)
        assert "1. English | level: 2" in mediawiki_sections("apple")

        mock_urlopen.return_value = _mock_urlopen(payload)
        result = wiktionary_entry("apple")
        assert "Wiktionary entry apple (English):" in result
        assert "Noun:" in result

    @patch("ai_arch_toolkit.toolkit.tools._mediawiki.urllib.request.urlopen")
    def test_invalid_options_do_not_call_api(self, mock_urlopen):
        assert "invalid api_url" in mediawiki_search("x", api_url="http://example.com/api.php")
        assert "invalid term" in wiktionary_entry("bad<>")
        mock_urlopen.assert_not_called()
