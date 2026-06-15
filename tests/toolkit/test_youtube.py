"""Tests for toolkit/tools/_youtube.py."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from ai_arch_toolkit.toolkit.tools._youtube import (
    youtube_transcript,
    youtube_transcript_languages,
    youtube_transcript_search,
)


class FakeYouTubeError(Exception):
    """Fake youtube-transcript-api base error."""


class FakeTranscript:
    def __init__(
        self,
        *,
        language_code: str = "en",
        language: str = "English",
        is_generated: bool = False,
        segments: list[SimpleNamespace] | None = None,
    ) -> None:
        self.language_code = language_code
        self.language = language
        self.is_generated = is_generated
        self.is_translatable = True
        self.translation_languages = [
            {"language_code": "pt", "language": "Portuguese"},
            {"language_code": "es", "language": "Spanish"},
        ]
        self._segments = segments or [
            SimpleNamespace(start=1.36, duration=1.68, text="[music]"),
            SimpleNamespace(start=18.64, duration=3.24, text="We're no strangers to love"),
            SimpleNamespace(start=22.64, duration=4.32, text="You know the rules and so do I"),
            SimpleNamespace(start=43.0, duration=2.12, text="Never gonna give you up"),
        ]

    def fetch(self, *, preserve_formatting: bool = False):
        assert preserve_formatting is False
        return self._segments

    def translate(self, language_code: str):
        return FakeTranscript(
            language_code=language_code,
            language="Portuguese",
            segments=[SimpleNamespace(start=1.0, duration=2.0, text="Texto traduzido")],
        )


_DEFAULT_MANUAL = object()


class FakeTranscriptList:
    def __init__(self, *, manual: FakeTranscript | None | object = _DEFAULT_MANUAL) -> None:
        self.manual = FakeTranscript() if manual is _DEFAULT_MANUAL else manual
        self.generated = FakeTranscript(language="English (auto-generated)", is_generated=True)

    def __iter__(self):
        transcripts = [self.generated] if self.manual is None else [self.manual, self.generated]
        return iter(transcripts)

    def find_manually_created_transcript(self, languages):
        if self.manual is None:
            raise FakeYouTubeError("manual transcript not found")
        return self.manual

    def find_generated_transcript(self, languages):
        return self.generated

    def find_transcript(self, languages):
        return self.manual or self.generated


class FakeYouTubeTranscriptApi:
    transcript_list = FakeTranscriptList()
    requested_video_id = ""

    def list(self, video_id: str):
        type(self).requested_video_id = video_id
        return type(self).transcript_list


def _fake_api():
    return FakeYouTubeTranscriptApi, FakeYouTubeError


class TestYouTubeTranscript:
    @patch("ai_arch_toolkit.toolkit.tools._youtube._load_youtube_transcript_api")
    def test_returns_transcript_text(self, mock_loader):
        mock_loader.return_value = _fake_api()

        result = youtube_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert result.startswith("YouTube transcript for dQw4w9WgXcQ:")
        assert "Language: en (English) | kind: manual" in result
        assert "We're no strangers to love" in result
        assert FakeYouTubeTranscriptApi.requested_video_id == "dQw4w9WgXcQ"

    @patch("ai_arch_toolkit.toolkit.tools._youtube._load_youtube_transcript_api")
    def test_returns_segments_and_translation(self, mock_loader):
        mock_loader.return_value = _fake_api()

        result = youtube_transcript("dQw4w9WgXcQ", translate_to="pt", output_format="segments")

        assert "Language: pt (Portuguese) | kind: manual" in result
        assert "[00:00:01.000 - 00:00:03.000] Texto traduzido" in result

    @patch("ai_arch_toolkit.toolkit.tools._youtube._load_youtube_transcript_api")
    def test_falls_back_to_generated_transcript(self, mock_loader):
        mock_loader.return_value = _fake_api()
        FakeYouTubeTranscriptApi.transcript_list = FakeTranscriptList(manual=None)

        try:
            result = youtube_transcript("dQw4w9WgXcQ", allow_generated=True)
        finally:
            FakeYouTubeTranscriptApi.transcript_list = FakeTranscriptList()

        assert "kind: generated" in result

    @patch("ai_arch_toolkit.toolkit.tools._youtube._load_youtube_transcript_api")
    def test_invalid_options_do_not_load_api(self, mock_loader):
        assert "invalid video URL or ID" in youtube_transcript("bad")
        mock_loader.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._youtube._load_youtube_transcript_api")
    def test_invalid_output_format_does_not_load_api(self, mock_loader):
        assert "output_format must be" in youtube_transcript("dQw4w9WgXcQ", output_format="xml")
        mock_loader.assert_not_called()

    @patch("ai_arch_toolkit.toolkit.tools._youtube._load_youtube_transcript_api")
    def test_missing_optional_dependency(self, mock_loader):
        mock_loader.return_value = (None, Exception)

        result = youtube_transcript("dQw4w9WgXcQ")

        assert "youtube-transcript-api is not installed" in result
        assert "uv sync --extra youtube" in result


class TestYouTubeTranscriptLanguages:
    @patch("ai_arch_toolkit.toolkit.tools._youtube._load_youtube_transcript_api")
    def test_lists_languages(self, mock_loader):
        mock_loader.return_value = _fake_api()

        result = youtube_transcript_languages("https://youtu.be/dQw4w9WgXcQ")

        assert result.startswith("YouTube transcript languages for dQw4w9WgXcQ:")
        assert "- en: English (manual, translatable)" in result
        assert "- en: English (auto-generated) (generated, translatable)" in result
        assert "translations: pt (Portuguese), es (Spanish)" in result


class TestYouTubeTranscriptSearch:
    @patch("ai_arch_toolkit.toolkit.tools._youtube._load_youtube_transcript_api")
    def test_search_returns_timestamped_matches(self, mock_loader):
        mock_loader.return_value = _fake_api()

        result = youtube_transcript_search(
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "never",
            context_segments=1,
        )

        assert result.startswith('YouTube transcript matches for "never" in dQw4w9WgXcQ:')
        assert "[00:00:22.640 - 00:00:45.120]" in result
        assert "You know the rules and so do I Never gonna give you up" in result

    @patch("ai_arch_toolkit.toolkit.tools._youtube._load_youtube_transcript_api")
    def test_search_handles_no_matches(self, mock_loader):
        mock_loader.return_value = _fake_api()

        result = youtube_transcript_search("dQw4w9WgXcQ", "missing")

        assert 'No matches found for "missing"' in result
