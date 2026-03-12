"""Tests for toolkit/tools/_dictionary.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ai_arch_toolkit.toolkit.tools._dictionary import define_word


def _mock_urlopen(data):
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestDefineWord:
    @patch("ai_arch_toolkit.toolkit.tools._dictionary.urllib.request.urlopen")
    def test_returns_definition(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
            [
                {
                    "word": "test",
                    "phonetic": "/tɛst/",
                    "meanings": [
                        {
                            "partOfSpeech": "noun",
                            "definitions": [{"definition": "A procedure for evaluation."}],
                        }
                    ],
                }
            ]
        )
        result = define_word("test")
        assert "test" in result
        assert "noun" in result
        assert "procedure" in result

    @patch("ai_arch_toolkit.toolkit.tools._dictionary.urllib.request.urlopen")
    def test_word_not_found(self, mock_urlopen):
        import urllib.error
        from io import BytesIO

        mock_urlopen.side_effect = urllib.error.HTTPError("url", 404, "Not Found", {}, BytesIO())
        result = define_word("xyzzzz")
        assert "not found" in result.lower()
