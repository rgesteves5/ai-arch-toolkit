"""Tests for toolkit/tools/_youtube.py."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._youtube import (
    youtube_transcript,
    youtube_transcript_languages,
    youtube_transcript_search,
)

yt = pytest.importorskip("youtube_transcript_api")
requests = pytest.importorskip("requests")

_LOADER = "ai_arch_toolkit.toolkit.tools._youtube._load_youtube_transcript_api"


def _failure(call):
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value.error


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
    def __init__(self, *, manual: FakeTranscript | object | None = _DEFAULT_MANUAL) -> None:
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

    @pytest.mark.parametrize(
        ("call", "words"),
        [
            (lambda: youtube_transcript("bad"), "invalid video URL or ID 'bad'"),
            (
                lambda: youtube_transcript("dQw4w9WgXcQ", output_format="xml"),
                "invalid output_format 'xml'",
            ),
            (lambda: youtube_transcript("dQw4w9WgXcQ", languages=" , "), "no language code"),
            (lambda: youtube_transcript_languages("bad"), "invalid video URL or ID"),
            (lambda: youtube_transcript_search("dQw4w9WgXcQ", "  "), "empty query"),
            (
                lambda: youtube_transcript_search("dQw4w9WgXcQ", "x", languages=""),
                "no language code",
            ),
        ],
    )
    @patch(_LOADER)
    def test_invalid_options_do_not_load_api(self, mock_loader, call, words):
        error = _failure(call)

        assert error.type == "validation_error"
        assert words in error.message
        mock_loader.assert_not_called()

    @patch(_LOADER)
    def test_missing_optional_dependency(self, mock_loader):
        mock_loader.return_value = (None, Exception)

        error = _failure(lambda: youtube_transcript("dQw4w9WgXcQ"))

        assert error.type == "upstream"
        assert "youtube-transcript-api is not installed" in error.message
        assert "uv sync --extra youtube" in error.message


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


class _OfflineApi:
    def list(self, video_id: str):
        raise ConnectionError("network is unreachable")


@pytest.mark.parametrize(
    "call",
    [
        lambda: youtube_transcript("dQw4w9WgXcQ"),
        lambda: youtube_transcript_languages("dQw4w9WgXcQ"),
        lambda: youtube_transcript_search("dQw4w9WgXcQ", "love"),
    ],
)
@patch(_LOADER)
def test_a_network_failure_is_a_retryable_upstream_failure(mock_loader, call):
    mock_loader.return_value = (_OfflineApi, FakeYouTubeError)

    error = _failure(call)

    assert error.type == "upstream"
    assert error.retryable
    assert "network is unreachable" in error.message


def _raising(error: Exception):
    """An API class whose ``list`` raises ``error``, with the library's real base error."""

    class _Api:
        def list(self, video_id: str):
            raise error

    return _Api, yt.YouTubeTranscriptApiException


def _http_error() -> Exception:
    return requests.HTTPError("500 Server Error")


@pytest.mark.parametrize(
    ("error", "kind", "words"),
    [
        (yt.NoTranscriptFound("dQw4w9WgXcQ", ["en"], None), "not_found", "no transcript in en"),
        (yt.TranscriptsDisabled("dQw4w9WgXcQ"), "not_found", "transcripts turned off"),
        (yt.VideoUnavailable("dQw4w9WgXcQ"), "not_found", "no YouTube video dQw4w9WgXcQ"),
        (yt.InvalidVideoId("dQw4w9WgXcQ"), "not_found", "check the URL or ID"),
        (yt.RequestBlocked("dQw4w9WgXcQ"), "rate_limited", "blocking requests from this IP"),
        (yt.IpBlocked("dQw4w9WgXcQ"), "rate_limited", "try again later"),
        (
            yt.YouTubeRequestFailed("dQw4w9WgXcQ", _http_error()),
            "upstream",
            "the request to YouTube failed: 500 Server Error",
        ),
        (yt.AgeRestricted("dQw4w9WgXcQ"), "upstream", "YouTube gave no transcript"),
    ],
)
@patch(_LOADER)
def test_each_library_error_has_its_type(mock_loader, error, kind, words):
    mock_loader.return_value = _raising(error)

    failure = _failure(lambda: youtube_transcript("dQw4w9WgXcQ"))

    assert failure.type == kind
    assert words in failure.message
    assert "github.com" not in failure.message  # the library's issue referral is left out


class _UntranslatableTranscript(FakeTranscript):
    def translate(self, language_code: str):
        raise yt.TranslationLanguageNotAvailable("dQw4w9WgXcQ")


@patch(_LOADER)
def test_a_translation_the_transcript_lacks_is_a_validation_error(mock_loader):
    class _Api:
        def list(self, video_id: str):
            return FakeTranscriptList(manual=_UntranslatableTranscript())

    mock_loader.return_value = (_Api, yt.YouTubeTranscriptApiException)

    failure = _failure(lambda: youtube_transcript("dQw4w9WgXcQ", translate_to="xx"))

    assert failure.type == "validation_error"
    assert "cannot be translated to 'xx'" in failure.message
    assert "youtube_transcript_languages" in failure.message


@patch(_LOADER)
def test_a_transcript_without_text_is_a_success(mock_loader):
    # FakeTranscript reads an empty list as "the default segments": give it one blank segment.
    blank = FakeTranscript(segments=[SimpleNamespace(start=0.0, duration=1.0, text="")])

    class _Api:
        def list(self, video_id: str):
            return FakeTranscriptList(manual=blank)

    mock_loader.return_value = (_Api, yt.YouTubeTranscriptApiException)

    assert youtube_transcript("dQw4w9WgXcQ") == (
        "The YouTube transcript of video dQw4w9WgXcQ has no text."
    )


@pytest.mark.parametrize(
    ("status", "retryable"), [("500 Server Error", True), ("403 Forbidden", False)]
)
@patch(_LOADER)
def test_only_a_server_error_from_youtube_is_retryable(mock_loader, status, retryable):
    mock_loader.return_value = _raising(
        yt.YouTubeRequestFailed("dQw4w9WgXcQ", requests.HTTPError(status))
    )

    failure = _failure(lambda: youtube_transcript("dQw4w9WgXcQ"))

    assert (failure.type, failure.retryable) == ("upstream", retryable)
