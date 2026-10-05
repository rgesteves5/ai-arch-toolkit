"""Tests for toolkit/tools/_dictionary.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._dictionary import define_word
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond


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
        with pytest.raises(ToolFailure) as caught:
            define_word("xyzzzz")

        assert caught.value.error.type == "not_found"
        assert "no entry for 'xyzzzz'" in caught.value.error.message
        assert "wiktionary_entry" in caught.value.error.message

    @patch(HTTP_OPEN)
    def test_no_definitions_is_a_success(self, mock_urlopen):
        mock_urlopen.return_value = respond([])

        assert define_word("test") == "No definitions found for: 'test'"

    @patch(HTTP_OPEN)
    def test_blank_word_does_not_call_api(self, mock_urlopen):
        with pytest.raises(ToolFailure) as caught:
            define_word("  ")

        assert caught.value.error.type == "validation_error"
        mock_urlopen.assert_not_called()

    @patch(HTTP_OPEN)
    def test_other_statuses_propagate(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(503, "Service Unavailable")

        with pytest.raises(ToolFailure) as caught:
            define_word("test")

        assert caught.value.error.type == "upstream"
        assert caught.value.error.retryable
        assert "HTTP error 503" in caught.value.error.message
