"""Tests for toolkit/tools/_dictionary.py."""

from __future__ import annotations

from unittest.mock import patch

from ai_arch_toolkit.toolkit.tools._dictionary import define_word
from tests.toolkit.http_fakes import HTTP_OPEN, respond


class TestDefineWord:
    @patch(HTTP_OPEN)
    def test_returns_definition(self, mock_urlopen):
        mock_urlopen.return_value = respond(
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

    @patch(HTTP_OPEN)
    def test_word_not_found(self, mock_urlopen):
        import urllib.error
        from io import BytesIO

        mock_urlopen.side_effect = urllib.error.HTTPError("url", 404, "Not Found", {}, BytesIO())
        result = define_word("xyzzzz")
        assert "not found" in result.lower()
