"""YouTube transcript tools, through youtube-transcript-api, read window by window (T09; D39).

The library is the source, not the HTTP door: it scrapes YouTube's watch page and its player API
(https://github.com/jdepoix/youtube-transcript-api), and says what went wrong by the class of the
exception it raises (``_errors.py``), which ``_transcript_failure`` types. A transcript is fetched
whole on every call; the window's footer names the call that reads on.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass, replace
from typing import Annotated, Any

from ai_arch_toolkit.core import Range, ToolResult, tool
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools._window import Window, page_window, text_window

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_MAX_CHARS = 50_000
_DEFAULT_MAX_CHARS = 12_000
_MAX_SEARCH_RESULTS = 20
# The languages list is short but for its translation targets (about a hundred on YouTube).
_LANGUAGES_CHARS = 4000
# What reading a page of an unexpected shape raises in the library: a missing key (its
# ``caption["name"]["runs"]``), malformed XML (``ElementTree.ParseError``, a ``SyntaxError``), a
# null where an object was. The door's ``_SHAPE_ERRORS``, for a source the door does not reach.
_SHAPE_ERRORS = (
    ArithmeticError,
    AttributeError,
    LookupError,
    RecursionError,
    SyntaxError,
    TypeError,
    ValueError,
)
_OPTIONAL_DEP_ERROR = (
    "youtube-transcript-api is not installed. Install the optional extra with "
    "`uv sync --extra youtube` or `pip install 'ai-arch-toolkit[youtube]'`."
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _TranscriptSegment:
    """Normalized YouTube transcript segment."""

    start: float
    duration: float
    text: str

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True, slots=True, kw_only=True)
class _TranscriptInfo:
    """Metadata for an available YouTube transcript."""

    language_code: str
    language: str
    is_generated: bool
    is_translatable: bool
    translation_languages: tuple[tuple[str, str], ...]


@tool(capability="network")
def youtube_transcript(
    video_url_or_id: str,
    languages: str = "en",
    prefer_manual: bool = True,
    allow_generated: bool = True,
    translate_to: str = "",
    output_format: str = "text",
    max_chars: Annotated[int, Range(1, _MAX_CHARS)] = _DEFAULT_MAX_CHARS,
    preserve_formatting: bool = False,
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """Fetch a public YouTube transcript.

    Args:
        video_url_or_id: YouTube video URL or 11-character video ID.
        languages: Comma-separated preferred source language codes, e.g. "en,pt-BR".
        prefer_manual: Prefer manually-created captions over auto-generated captions.
        allow_generated: Allow auto-generated captions if manual captions are unavailable.
        translate_to: Optional target language code supported by the transcript.
        output_format: One of "text", "segments", "json", "srt", or "vtt".
        max_chars: How many characters of the transcript to return.
        preserve_formatting: Preserve HTML formatting where supported by the provider.
        offset: Where to start, in characters of the transcript; the footer gives the next
            offset.

    Raises:
        ToolFailure: validation_error when the video, the languages, the output format or the
            translation target is invalid; not_found when the video or a transcript in those
            languages does not exist; rate_limited when YouTube blocks the requests; upstream
            when YouTube fails or youtube-transcript-api is not installed.
    """
    video_id = _video_id(video_url_or_id)
    language_codes = _languages(languages)

    output_format = output_format.strip().lower()
    if output_format not in {"text", "segments", "json", "srt", "vtt"}:
        raise ToolFailure(
            "validation_error",
            f"invalid output_format {output_format!r}; use text, segments, json, srt, or vtt",
        )

    transcript, segments, source = _fetch_transcript(
        video_id,
        language_codes,
        prefer_manual=prefer_manual,
        allow_generated=allow_generated,
        translate_to=translate_to.strip(),
        preserve_formatting=preserve_formatting,
    )

    if not segments:
        return ToolResult.success(f"The YouTube transcript of video {video_id} has no text.")

    body = _format_segments(segments, output_format)
    window = text_window(body, offset=offset, limit=max_chars)
    return window.result(heading=f"{_describe(video_id, transcript, source)}:")


@tool(capability="network")
def youtube_transcript_languages(
    video_url_or_id: str, offset: Annotated[int, Range(0)] = 0
) -> ToolResult:
    """List a YouTube video's public transcripts and the languages they translate to.

    Args:
        video_url_or_id: YouTube video URL or 11-character video ID.
        offset: Where to start, in characters of the list; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when the video URL or ID is malformed; not_found when the
            video does not exist or has transcripts turned off; rate_limited when YouTube blocks
            the requests; upstream when YouTube fails or youtube-transcript-api is not installed.
    """
    video_id = _video_id(video_url_or_id)
    api_cls, error_cls = _api()

    try:
        infos = [_transcript_info(transcript) for transcript in api_cls().list(video_id)]
    except error_cls as e:
        raise _transcript_failure(e, video_id) from e
    except OSError as e:  # requests' network errors
        raise _network_failure(e) from e
    except _SHAPE_ERRORS as e:
        raise ToolFailure("upstream", f"could not parse the transcript list: {e!r}") from e

    if not infos:
        return ToolResult.success(f"No YouTube transcripts found for video {video_id}.")

    window = text_window(_languages_text(infos), offset=offset, limit=_LANGUAGES_CHARS)
    return window.result(heading=f"YouTube transcripts of {video_id}:")


@tool(capability="network")
def youtube_transcript_search(
    video_url_or_id: str,
    query: str,
    languages: str = "en",
    prefer_manual: bool = True,
    allow_generated: bool = True,
    max_results: Annotated[int, Range(1, _MAX_SEARCH_RESULTS)] = 10,
    context_segments: Annotated[int, Range(0, 3)] = 1,
    preserve_formatting: bool = False,
    offset: Annotated[int, Range(0)] = 0,
) -> ToolResult:
    """Search within a public YouTube transcript and return timestamped matches.

    Args:
        video_url_or_id: YouTube video URL or 11-character video ID.
        query: Case-insensitive text to find in transcript segments.
        languages: Comma-separated preferred source language codes, e.g. "en,pt-BR".
        prefer_manual: Prefer manually-created captions over auto-generated captions.
        allow_generated: Allow auto-generated captions if manual captions are unavailable.
        max_results: How many matches to return.
        context_segments: How many neighboring segments to show on each side of a match.
        preserve_formatting: Preserve HTML formatting where supported by the provider.
        offset: How many matches to skip; the footer gives the next offset.

    Raises:
        ToolFailure: validation_error when the video, the query or the languages are invalid;
            not_found when the video or a transcript in those languages does not exist;
            rate_limited when YouTube blocks the requests; upstream when YouTube fails or
            youtube-transcript-api is not installed.
    """
    video_id = _video_id(video_url_or_id)
    if not query.strip():
        raise ToolFailure("validation_error", "empty query; give the text to find")
    language_codes = _languages(languages)

    transcript, segments, _ = _fetch_transcript(
        video_id,
        language_codes,
        prefer_manual=prefer_manual,
        allow_generated=allow_generated,
        translate_to="",
        preserve_formatting=preserve_formatting,
    )

    where = _describe(video_id, transcript, "")
    needle = query.casefold()
    matches = [
        index for index, segment in enumerate(segments) if needle in segment.text.casefold()
    ]
    if not matches:
        return ToolResult.success(f"The {where}, has no passage that mentions {query!r}.")

    lines = [
        f"{number}. {_passage(segments, index, context_segments)}"
        for number, index in enumerate(matches, start=1)
    ]
    window = _matches(lines, query, offset=offset, limit=max_results)
    return window.result(heading=f"{where}: passages that mention {query!r}:")


def _passage(segments: list[_TranscriptSegment], index: int, context: int) -> str:
    """The match at ``index`` with ``context`` segments on each side, timestamped (the match
    alone when ``context`` leaves nothing around it)."""
    shown = segments[max(0, index - context) : index + context + 1] or segments[index : index + 1]
    start_time = _timestamp(shown[0].start, decimal=True)
    end_time = _timestamp(shown[-1].end, decimal=True)
    text = " ".join(segment.text.replace("\n", " ").strip() for segment in shown)
    return f"[{start_time} - {end_time}] {text}"


def _matches(lines: list[str], query: str, *, offset: int, limit: int) -> Window:
    """A page of the match lines, counted as matches of ``query``."""
    page = page_window(lines, offset=offset, limit=limit)
    if not page.body:  # past the last match
        return Window(body="", unit="matches", first=0, last=0, total=len(lines), label=query)
    return replace(page, unit="matches", label=query)


def _describe(video_id: str, transcript: Any, translated_from: str) -> str:
    """Which transcript was read: the video, the language with its code, and the kind."""
    kind = "generated" if bool(_attr(transcript, "is_generated")) else "manual"
    code, language = str(_attr(transcript, "language_code")), str(_attr(transcript, "language"))
    described = f"YouTube transcript of {video_id}, {code} ({language}), {kind}"
    return described + (f", translated from {translated_from}" if translated_from else "")


def _languages_text(infos: list[_TranscriptInfo]) -> str:
    """The transcripts, one a line, then every language they translate to (one list for a
    video: YouTube offers the same targets for each transcript that translates)."""
    lines = []
    for info in infos:
        kind = "generated" if info.is_generated else "manual"
        translatable = "translatable" if info.is_translatable else "not translatable"
        lines.append(f"- {info.language_code}: {info.language} ({kind}, {translatable})")
    targets = dict(pair for info in infos for pair in info.translation_languages)
    if targets:
        lines.append("Translations (youtube_transcript translate_to=...):")
        lines += [f"- {code}: {language}" for code, language in targets.items()]
    return "\n".join(lines)


def _load_youtube_transcript_api() -> tuple[Any | None, type[Exception]]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import YouTubeTranscriptApiException
    except ImportError:
        return None, Exception
    return YouTubeTranscriptApi, YouTubeTranscriptApiException


def _api() -> tuple[Any, type[Exception]]:
    """The API class and its base error; without the optional extra, raises ``ToolFailure``."""
    api_cls, error_cls = _load_youtube_transcript_api()
    if api_cls is None:
        raise ToolFailure("upstream", _OPTIONAL_DEP_ERROR)
    return api_cls, error_cls


def _fetch_transcript(
    video_id: str,
    languages: tuple[str, ...],
    *,
    prefer_manual: bool,
    allow_generated: bool,
    translate_to: str,
    preserve_formatting: bool,
) -> tuple[Any, list[_TranscriptSegment], str]:
    """The chosen transcript (translated when asked), its segments, and the language code it
    was translated from (empty when it was not).

    Raises:
        ToolFailure: For every error youtube-transcript-api or the network raises.
    """
    api_cls, error_cls = _api()
    try:
        transcript = _select_transcript(
            api_cls().list(video_id),
            languages,
            prefer_manual=prefer_manual,
            allow_generated=allow_generated,
        )
        source = str(_attr(transcript, "language_code")) if translate_to else ""
        if translate_to:
            transcript = transcript.translate(translate_to)
        segments = _segments(transcript.fetch(preserve_formatting=preserve_formatting))
    except error_cls as e:
        raise _transcript_failure(e, video_id, languages, translate_to) from e
    except OSError as e:  # requests' network errors
        raise _network_failure(e) from e
    except _SHAPE_ERRORS as e:
        raise ToolFailure("upstream", f"could not parse the transcript response: {e!r}") from e
    return transcript, segments, source


def _transcript_failure(
    error: Exception, video_id: str, languages: tuple[str, ...] = (), translate_to: str = ""
) -> ToolFailure:
    """The typed failure for an error youtube-transcript-api raised.

    The library's exceptions name the cause
    (https://github.com/jdepoix/youtube-transcript-api/blob/master/youtube_transcript_api/_errors.py):
    a missing video or transcript is ``not_found``, a translation target the transcript does not
    offer is a ``validation_error``, YouTube blocking the IP is ``rate_limited``, and the rest
    is ``upstream``.
    """
    try:
        import youtube_transcript_api as yt
    except ImportError:  # pragma: no cover - the error came from the library
        return ToolFailure("upstream", str(error).strip())

    listing = "list the video's transcripts with youtube_transcript_languages"
    if isinstance(error, yt.NoTranscriptFound):
        wanted = ", ".join(languages) or "the requested languages"
        return ToolFailure(
            "not_found", f"video {video_id} has no transcript in {wanted}; {listing}"
        )
    if isinstance(error, yt.TranscriptsDisabled):
        return ToolFailure("not_found", f"video {video_id} has its transcripts turned off")
    if isinstance(error, yt.VideoUnavailable | yt.InvalidVideoId):
        return ToolFailure(
            "not_found", f"no YouTube video {video_id} is available; check the URL or ID"
        )
    if isinstance(error, yt.TranslationLanguageNotAvailable | yt.NotTranslatable):
        return ToolFailure(
            "validation_error",
            f"the transcript of video {video_id} cannot be translated to {translate_to!r}; "
            f"{listing} and their translations",
        )
    if isinstance(error, yt.RequestBlocked):
        return ToolFailure(
            "rate_limited",
            "YouTube is blocking requests from this IP (too many requests, or a cloud "
            "provider's address); try again later or from another network",
            retryable=True,
        )
    if isinstance(error, yt.YouTubeRequestFailed):  # the reason starts with the HTTP status
        server = error.reason[:1] == "5"
        return ToolFailure(
            "upstream", f"the request to YouTube failed: {error.reason}", retryable=server
        )
    if isinstance(error, yt.CouldNotRetrieveTranscript):
        cause = error.cause.strip().split("\n\n")[0] or "no reason given"
        return ToolFailure("upstream", f"YouTube gave no transcript of video {video_id}: {cause}")
    return ToolFailure("upstream", str(error).strip() or type(error).__name__)


def _network_failure(error: OSError) -> ToolFailure:
    return ToolFailure("upstream", f"network error reaching YouTube: {error}", retryable=True)


def _video_id(video_url_or_id: str) -> str:
    """The 11-character video ID; a value that holds none raises ``ToolFailure``."""
    video_id = _normalize_video_id(video_url_or_id)
    if not video_id:
        raise ToolFailure(
            "validation_error",
            f"invalid video URL or ID {video_url_or_id!r}; give a youtube.com or youtu.be URL "
            "or an 11-character ID such as dQw4w9WgXcQ",
        )
    return video_id


def _languages(languages: str) -> tuple[str, ...]:
    """The language codes; none raises ``ToolFailure`` (validation_error)."""
    codes = _parse_languages(languages)
    if not codes:
        raise ToolFailure(
            "validation_error", "no language code given; use codes such as 'en' or 'en,pt-BR'"
        )
    return codes


def _select_transcript(
    transcript_list: Any,
    languages: tuple[str, ...],
    *,
    prefer_manual: bool,
    allow_generated: bool,
) -> Any:
    selectors: list[str] = []
    if prefer_manual:
        selectors.append("manual")
        if allow_generated:
            selectors.append("generated")
    else:
        if allow_generated:
            selectors.append("generated")
        selectors.append("manual")

    last_error: Exception | None = None
    for selector in selectors:
        try:
            if selector == "manual":
                return transcript_list.find_manually_created_transcript(languages)
            return transcript_list.find_generated_transcript(languages)
        except Exception as e:  # youtube-transcript-api raises provider-specific subclasses.
            last_error = e

    if last_error is not None:
        raise last_error
    return transcript_list.find_transcript(languages)


def _normalize_video_id(value: str) -> str:
    value = value.strip()
    if _VIDEO_ID_RE.fullmatch(value):
        return value

    parsed = urllib.parse.urlparse(value)
    host = (parsed.netloc or "").lower()
    path_parts = [part for part in parsed.path.split("/") if part]
    query = urllib.parse.parse_qs(parsed.query)

    if "youtu.be" in host and path_parts and _VIDEO_ID_RE.fullmatch(path_parts[0]):
        return path_parts[0]
    if "youtube.com" in host or "youtube-nocookie.com" in host:
        video_id = query.get("v", [""])[0]
        if _VIDEO_ID_RE.fullmatch(video_id):
            return video_id
        if path_parts and path_parts[0] in {"embed", "shorts", "live", "v"}:
            candidate = path_parts[1] if len(path_parts) > 1 else ""
            if _VIDEO_ID_RE.fullmatch(candidate):
                return candidate
    return ""


def _parse_languages(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _segments(transcript: Any) -> list[_TranscriptSegment]:
    segments: list[_TranscriptSegment] = []
    for item in transcript:
        text = _attr(item, "text")
        if not text:
            continue
        segments.append(
            _TranscriptSegment(
                start=float(_attr(item, "start", 0.0) or 0.0),
                duration=float(_attr(item, "duration", 0.0) or 0.0),
                text=str(text).strip(),
            )
        )
    return segments


def _transcript_info(transcript: Any) -> _TranscriptInfo:
    translations: list[tuple[str, str]] = []
    for item in _attr(transcript, "translation_languages", []) or []:
        code = str(
            item.get("language_code", "")
            if isinstance(item, dict)
            else _attr(item, "language_code")
        )
        language = str(
            item.get("language", "") if isinstance(item, dict) else _attr(item, "language")
        )
        if code or language:
            translations.append((code, language))
    return _TranscriptInfo(
        language_code=str(_attr(transcript, "language_code")),
        language=str(_attr(transcript, "language")),
        is_generated=bool(_attr(transcript, "is_generated")),
        is_translatable=bool(_attr(transcript, "is_translatable")),
        translation_languages=tuple(translations),
    )


def _attr(value: Any, name: str, default: Any = "") -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _format_segments(segments: list[_TranscriptSegment], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(
            [
                {
                    "start": segment.start,
                    "duration": segment.duration,
                    "end": segment.end,
                    "text": segment.text,
                }
                for segment in segments
            ],
            ensure_ascii=False,
            indent=2,
        )
    if output_format == "segments":
        return "\n".join(
            f"[{_timestamp(segment.start, decimal=True)} - "
            f"{_timestamp(segment.end, decimal=True)}] {segment.text}"
            for segment in segments
        )
    if output_format == "srt":
        blocks = []
        for index, segment in enumerate(segments, start=1):
            blocks.append(
                f"{index}\n"
                f"{_srt_timestamp(segment.start)} --> {_srt_timestamp(segment.end)}\n"
                f"{segment.text}"
            )
        return "\n\n".join(blocks)
    if output_format == "vtt":
        blocks = ["WEBVTT"]
        for segment in segments:
            blocks.append(
                f"{_vtt_timestamp(segment.start)} --> {_vtt_timestamp(segment.end)}\n"
                f"{segment.text}"
            )
        return "\n\n".join(blocks)
    return "\n".join(segment.text for segment in segments)


def _timestamp(seconds: float, *, decimal: bool) -> str:
    millis = round(seconds * 1000)
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    if decimal:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _srt_timestamp(seconds: float) -> str:
    return _timestamp(seconds, decimal=True).replace(".", ",")


def _vtt_timestamp(seconds: float) -> str:
    return _timestamp(seconds, decimal=True)
