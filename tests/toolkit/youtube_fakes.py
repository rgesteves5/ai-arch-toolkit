"""Stand-ins for ``youtube-transcript-api``, the ``youtube_*`` tools' source (T09).

The tools reach the library through one seam, ``_youtube._load_youtube_transcript_api()``, which
returns its API class and its base error. Patch ``LOADER`` with :func:`library` (an API whose
``list`` answers a :class:`FakeTranscriptList`) or :func:`raising` (one whose ``list`` raises one
of the library's own errors, by name). The library is imported only when an error is raised, so
this module loads without the ``youtube`` extra.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any, ClassVar

LOADER = "ai_arch_toolkit.toolkit.tools._youtube._load_youtube_transcript_api"

type Loader = Callable[[], tuple[type[Any], type[Exception]]]


class FakeError(Exception):
    """The base error of a fake library."""


def segment(start: float, duration: float, text: str) -> SimpleNamespace:
    """One snippet of a fetched transcript, as the library gives it."""
    return SimpleNamespace(start=start, duration=duration, text=text)


SONG = (
    segment(1.36, 1.68, "[music]"),
    segment(18.64, 3.24, "We're no strangers to love"),
    segment(22.64, 4.32, "You know the rules and so do I"),
    segment(43.0, 2.12, "Never gonna give you up"),
)
TRANSLATIONS = (("pt", "Portuguese"), ("es", "Spanish"))


class FakeTranscript:
    """A transcript of the list: its language, its kind and its snippets."""

    def __init__(
        self,
        *,
        language_code: str = "en",
        language: str = "English",
        is_generated: bool = False,
        segments: Sequence[SimpleNamespace] = SONG,
        translations: Sequence[tuple[str, str]] = TRANSLATIONS,
    ) -> None:
        self.language_code = language_code
        self.language = language
        self.is_generated = is_generated
        self.is_translatable = bool(translations)
        self.translation_languages = [
            {"language_code": code, "language": name} for code, name in translations
        ]
        self._segments = list(segments)

    def fetch(self, *, preserve_formatting: bool = False) -> list[SimpleNamespace]:
        return self._segments

    def translate(self, language_code: str) -> FakeTranscript:
        return FakeTranscript(
            language_code=language_code,
            language="Portuguese",
            segments=[segment(1.0, 2.0, "Texto traduzido")],
            translations=(),
        )


class FakeTranscriptList:
    """A video's transcripts: a manual one (unless ``manual`` is ``None``) and a generated one."""

    def __init__(
        self,
        *,
        manual: FakeTranscript | None = None,
        generated: FakeTranscript | None = None,
        without_manual: bool = False,
    ) -> None:
        self.manual = None if without_manual else manual or FakeTranscript()
        self.generated = generated or FakeTranscript(
            language="English (auto-generated)", is_generated=True
        )

    def __iter__(self):
        return iter([t for t in (self.manual, self.generated) if t is not None])

    def find_manually_created_transcript(self, languages: Sequence[str]) -> FakeTranscript:
        if self.manual is None:
            raise FakeError("manual transcript not found")
        return self.manual

    def find_generated_transcript(self, languages: Sequence[str]) -> FakeTranscript:
        return self.generated

    def find_transcript(self, languages: Sequence[str]) -> FakeTranscript:
        return self.manual or self.generated


def library(transcripts: FakeTranscriptList | None = None) -> Loader:
    """A loader whose API lists ``transcripts`` for any video, recording the IDs asked for."""
    answer = transcripts or FakeTranscriptList()

    class _Api:
        asked: ClassVar[list[str]] = []

        def list(self, video_id: str) -> FakeTranscriptList:
            type(self).asked.append(video_id)
            return answer

    return lambda: (_Api, FakeError)


def raising(error: str, *args: Any) -> Loader:
    """A loader whose API raises the library's error ``error`` (``VideoUnavailable`` …), built
    with the video ID and ``args``, with the library's real base error."""

    def load() -> tuple[type[Any], type[Exception]]:
        import youtube_transcript_api as yt

        class _Api:
            def list(self, video_id: str) -> FakeTranscriptList:
                raise getattr(yt, error)(video_id, *args)

        return _Api, yt.YouTubeTranscriptApiException

    return load


def lines(count: int, word: str = "line") -> list[SimpleNamespace]:
    """``count`` snippets two seconds apart, each ``"{word} N"``."""
    return [segment(2.0 * n, 2.0, f"{word} {n}") for n in range(count)]
