"""Tests for toolkit/tools/_youtube.py: transcripts read window by window (T09).

The library is replaced at the module's one seam (``youtube_fakes.LOADER``).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._youtube import (
    youtube_transcript,
    youtube_transcript_languages,
    youtube_transcript_search,
)
from tests.toolkit.youtube_fakes import (
    LOADER,
    FakeTranscript,
    FakeTranscriptList,
    library,
    lines,
    raising,
    segment,
)

yt = pytest.importorskip("youtube_transcript_api")
requests = pytest.importorskip("requests")


def _failure(call):
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value.error


def _text(result: ToolResult | str) -> str:
    return result.value if isinstance(result, ToolResult) else result


def _window(result: ToolResult | str) -> dict[str, Any]:
    assert isinstance(result, ToolResult)
    return result.metadata["window"]


def _with(**transcript: Any) -> FakeTranscriptList:
    return FakeTranscriptList(manual=FakeTranscript(**transcript))


class TestYouTubeTranscript:
    def test_returns_transcript_text(self):
        load = library()
        with patch(LOADER, load):
            result = youtube_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert _text(result) == (
            "YouTube transcript of dQw4w9WgXcQ, en (English), manual:\n"
            "[music]\nWe're no strangers to love\nYou know the rules and so do I\n"
            "Never gonna give you up"
        )
        assert load()[0].asked == ["dQw4w9WgXcQ"]

    @patch(LOADER, library())
    def test_returns_segments_and_translation(self):
        result = _text(
            youtube_transcript("dQw4w9WgXcQ", translate_to="pt", output_format="segments")
        )

        assert result.startswith(
            "YouTube transcript of dQw4w9WgXcQ, pt (Portuguese), manual, translated from en:\n"
        )
        assert "[00:00:01.000 - 00:00:03.000] Texto traduzido" in result

    @patch(LOADER, library(FakeTranscriptList(without_manual=True)))
    def test_falls_back_to_generated_transcript(self):
        result = _text(youtube_transcript("dQw4w9WgXcQ", allow_generated=True))

        assert result.startswith(
            "YouTube transcript of dQw4w9WgXcQ, en (English (auto-generated)), generated:"
        )

    @patch(LOADER, library(_with(segments=lines(400))))
    def test_following_the_footers_rebuilds_the_whole_transcript(self):
        whole = "\n".join(f"line {n}" for n in range(400))
        parts: list[str] = []
        offset: int | None = 0

        while offset is not None:
            result = youtube_transcript("dQw4w9WgXcQ", max_chars=500, offset=offset)
            window = _window(result)
            body = _text(result).split("\n", 1)[1]  # under the heading
            parts.append(body[: window["last"] - window["first"]])
            offset = (window["next_call"] or {}).get("offset")

        assert "".join(parts) == whole
        assert len(parts) > 5

    @patch(LOADER, library(_with(segments=lines(10_000))))
    def test_a_cut_at_the_ceiling_names_the_next_offset_not_a_larger_max_chars(self):
        # It said "Increase max_chars" even at the 50,000 ceiling.
        text = _text(result := youtube_transcript("dQw4w9WgXcQ", max_chars=50_000))

        window = _window(result)
        assert window["total"] == len("\n".join(f"line {n}" for n in range(10_000)))
        assert text.endswith(
            f"[chars 0-{window['last']} of {window['total']} | next: offset={window['last']}]"
        )
        assert "max_chars" not in text

    @pytest.mark.parametrize("output_format", ["json", "srt", "vtt"])
    @patch(LOADER, library(_with(segments=lines(300))))
    def test_every_format_reads_window_by_window(self, output_format):
        first = youtube_transcript("dQw4w9WgXcQ", output_format=output_format, max_chars=400)
        onward = _window(first)["next_call"]
        second = youtube_transcript(
            "dQw4w9WgXcQ", output_format=output_format, max_chars=400, **onward
        )

        assert _window(second)["first"] == _window(first)["last"]

    @pytest.mark.parametrize(
        ("max_chars", "kept"), [(1, True), (50_000, True), (0, False), (50_001, False)]
    )
    @patch(LOADER, library())
    def test_max_chars_is_refused_outside_its_limits_through_the_executor(self, max_chars, kept):
        call = ToolCall(
            id="c1",
            name="youtube_transcript",
            input={"video_url_or_id": "dQw4w9WgXcQ", "max_chars": max_chars},
        )

        result = ToolGroup(youtube_transcript).execute(call)

        assert result.ok is kept
        if not kept:
            assert result.error is not None
            assert result.error.type == "validation_error"

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
    @patch(LOADER)
    def test_invalid_options_do_not_load_api(self, mock_loader, call, words):
        error = _failure(call)

        assert error.type == "validation_error"
        assert words in error.message
        mock_loader.assert_not_called()

    @patch(LOADER)
    def test_missing_optional_dependency(self, mock_loader):
        mock_loader.return_value = (None, Exception)

        error = _failure(lambda: youtube_transcript("dQw4w9WgXcQ"))

        assert error.type == "upstream"
        assert "youtube-transcript-api is not installed" in error.message
        assert "uv sync --extra youtube" in error.message


class TestYouTubeTranscriptLanguages:
    @patch(LOADER, library())
    def test_lists_languages(self):
        result = _text(youtube_transcript_languages("https://youtu.be/dQw4w9WgXcQ"))

        assert result == (
            "YouTube transcripts of dQw4w9WgXcQ:\n"
            "- en: English (manual, translatable)\n"
            "- en: English (auto-generated) (generated, translatable)\n"
            "Translations (youtube_transcript translate_to=...):\n"
            "- pt: Portuguese\n"
            "- es: Spanish"
        )

    @patch(
        LOADER,
        library(_with(translations=[(f"l{n}", f"Language number {n}") for n in range(300)])),
    )
    def test_every_translation_is_listed_window_by_window(self):
        # They were cut at eight, with "+N more" and no way to read the rest.
        seen: list[str] = []
        offset: int | None = 0

        while offset is not None:
            result = youtube_transcript_languages("dQw4w9WgXcQ", offset=offset)
            seen += [line for line in _text(result).splitlines() if line.startswith("- l")]
            offset = (_window(result)["next_call"] or {}).get("offset")

        assert seen == [f"- l{n}: Language number {n}" for n in range(300)]

    @patch(LOADER, library(FakeTranscriptList(manual=FakeTranscript(translations=()))))
    def test_a_transcript_that_cannot_be_translated_says_so(self):
        result = _text(youtube_transcript_languages("dQw4w9WgXcQ"))

        assert "- en: English (manual, not translatable)" in result


class TestYouTubeTranscriptSearch:
    @patch(LOADER, library())
    def test_search_returns_timestamped_matches(self):
        result = _text(
            youtube_transcript_search(
                "https://www.youtube.com/shorts/dQw4w9WgXcQ", "never", context_segments=1
            )
        )

        assert result == (
            "YouTube transcript of dQw4w9WgXcQ, en (English), manual: passages that mention "
            "'never':\n"
            "1. [00:00:22.640 - 00:00:45.120] You know the rules and so do I Never gonna give "
            'you up\n[matches 1-1 of 1 for "never" | end]'
        )

    @patch(LOADER, library(_with(segments=lines(50, word="chorus"))))
    def test_matches_page_by_offset_with_their_total(self):
        # They ended in "... N more matches not shown", with no way to read them.
        first = youtube_transcript_search("dQw4w9WgXcQ", "chorus", max_results=20)
        second = youtube_transcript_search(
            "dQw4w9WgXcQ", "chorus", max_results=20, **_window(first)["next_call"]
        )
        third = youtube_transcript_search("dQw4w9WgXcQ", "chorus", max_results=20, offset=40)

        assert _text(first).endswith('[matches 1-20 of 50 for "chorus" | next: offset=20]')
        assert _text(second).splitlines()[1].startswith("21. [00:00:38.000 - ")
        assert _text(second).endswith('[matches 21-40 of 50 for "chorus" | next: offset=40]')
        assert _text(third).endswith('[matches 41-50 of 50 for "chorus" | end]')

    @patch(LOADER, library())
    def test_a_raw_call_with_a_negative_context_shows_the_match_alone(self):
        # The executor refuses it (Range); called raw, the tool still answers.
        text = _text(youtube_transcript_search("dQw4w9WgXcQ", "never", context_segments=-1))

        assert "1. [00:00:43.000 - 00:00:45.120] Never gonna give you up" in text

    @patch(LOADER, library())
    def test_an_offset_past_the_last_match_says_so(self):
        text = _text(youtube_transcript_search("dQw4w9WgXcQ", "never", offset=5))

        assert text.endswith('[no more matches for "never" of 1 | end]')

    @patch(LOADER, library())
    def test_search_handles_no_matches(self):
        result = _text(youtube_transcript_search("dQw4w9WgXcQ", "missing"))

        assert result == (
            "No passages of the YouTube transcript of dQw4w9WgXcQ, en (English), manual, "
            "mention 'missing'."
        )


class _OfflineApi:
    def list(self, video_id: str):
        raise ConnectionError("network is unreachable")


class FakeYouTubeError(Exception):
    """A base error no other exception derives from."""


@pytest.mark.parametrize(
    "call",
    [
        lambda: youtube_transcript("dQw4w9WgXcQ"),
        lambda: youtube_transcript_languages("dQw4w9WgXcQ"),
        lambda: youtube_transcript_search("dQw4w9WgXcQ", "love"),
    ],
)
@patch(LOADER)
def test_a_network_failure_is_a_retryable_upstream_failure(mock_loader, call):
    mock_loader.return_value = (_OfflineApi, FakeYouTubeError)

    error = _failure(call)

    assert error.type == "upstream"
    assert error.retryable
    assert "network is unreachable" in error.message


@pytest.mark.parametrize(
    ("error", "args", "kind", "words"),
    [
        ("NoTranscriptFound", (["en"], None), "not_found", "no transcript in en"),
        ("TranscriptsDisabled", (), "not_found", "transcripts turned off"),
        ("VideoUnavailable", (), "not_found", "no YouTube video dQw4w9WgXcQ"),
        ("InvalidVideoId", (), "not_found", "check the URL or ID"),
        ("RequestBlocked", (), "rate_limited", "blocking requests from this IP"),
        ("IpBlocked", (), "rate_limited", "try again later"),
        (
            "YouTubeRequestFailed",
            (requests.HTTPError("500 Server Error"),),
            "upstream",
            "the request to YouTube failed: 500 Server Error",
        ),
        ("AgeRestricted", (), "upstream", "YouTube gave no transcript"),
    ],
)
def test_each_library_error_has_its_type(error, args, kind, words):
    with patch(LOADER, raising(error, *args)):
        failure = _failure(lambda: youtube_transcript("dQw4w9WgXcQ"))

    assert failure.type == kind
    assert words in failure.message
    assert "github.com" not in failure.message  # the library's issue referral is left out


@pytest.mark.parametrize(
    "call",
    [
        lambda: youtube_transcript("dQw4w9WgXcQ"),
        lambda: youtube_transcript_languages("dQw4w9WgXcQ"),
        lambda: youtube_transcript_search("dQw4w9WgXcQ", "love"),
    ],
)
def test_a_video_that_is_not_there_is_not_found_for_every_tool(call):
    with patch(LOADER, raising("VideoUnavailable")):
        failure = _failure(call)

    assert failure.type == "not_found"
    assert "check the URL or ID" in failure.message


class _UntranslatableTranscript(FakeTranscript):
    def translate(self, language_code: str):
        raise yt.TranslationLanguageNotAvailable("dQw4w9WgXcQ")


def test_a_translation_the_transcript_lacks_is_a_validation_error():
    class _Api:
        def list(self, video_id: str):
            return FakeTranscriptList(manual=_UntranslatableTranscript())

    with patch(LOADER, lambda: (_Api, yt.YouTubeTranscriptApiException)):
        failure = _failure(lambda: youtube_transcript("dQw4w9WgXcQ", translate_to="xx"))

    assert failure.type == "validation_error"
    assert "cannot be translated to 'xx'" in failure.message
    assert "youtube_transcript_languages" in failure.message


@patch(LOADER, library(_with(segments=[segment(0.0, 1.0, "")])))
def test_a_transcript_without_text_is_a_success():
    assert _text(youtube_transcript("dQw4w9WgXcQ")) == (
        "The YouTube transcript of video dQw4w9WgXcQ has no text."
    )


@pytest.mark.parametrize(
    ("status", "retryable"), [("500 Server Error", True), ("403 Forbidden", False)]
)
def test_only_a_server_error_from_youtube_is_retryable(status, retryable):
    with patch(LOADER, raising("YouTubeRequestFailed", requests.HTTPError(status))):
        failure = _failure(lambda: youtube_transcript("dQw4w9WgXcQ"))

    assert (failure.type, failure.retryable) == ("upstream", retryable)
