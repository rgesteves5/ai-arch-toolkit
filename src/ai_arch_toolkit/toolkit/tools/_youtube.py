"""YouTube transcript tools powered by youtube-transcript-api."""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core import tool
from ai_arch_toolkit.core._tools._result import ToolFailure

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_MAX_CHARS_LIMIT = 50_000
_DEFAULT_MAX_CHARS = 12_000
_MAX_SEARCH_RESULTS = 20
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
    max_chars: int = _DEFAULT_MAX_CHARS,
    preserve_formatting: bool = False,
) -> str:
    """Fetch a public YouTube transcript.

    Args:
        video_url_or_id: YouTube video URL or 11-character video ID.
        languages: Comma-separated preferred source language codes, e.g. "en,pt-BR".
        prefer_manual: Prefer manually-created captions over auto-generated captions.
        allow_generated: Allow auto-generated captions if manual captions are unavailable.
        translate_to: Optional target language code supported by the transcript.
        output_format: One of "text", "segments", "json", "srt", or "vtt".
        max_chars: Maximum output characters (1-50000). Defaults to 12000.
        preserve_formatting: Preserve HTML formatting where supported by the provider.

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

    max_chars = _clamp(max_chars, 1, _MAX_CHARS_LIMIT)
    transcript, segments = _fetch_transcript(
        video_id,
        language_codes,
        prefer_manual=prefer_manual,
        allow_generated=allow_generated,
        translate_to=translate_to.strip(),
        preserve_formatting=preserve_formatting,
    )

    if not segments:
        return f"The YouTube transcript of video {video_id} has no text."

    header = _transcript_header(video_id, transcript)
    body = _format_segments(segments, output_format)
    return _limit_text(f"{header}\n{body}", max_chars)


@tool(capability="network")
def youtube_transcript_languages(video_url_or_id: str) -> str:
    """List public transcript languages available for a YouTube video.

    Args:
        video_url_or_id: YouTube video URL or 11-character video ID.

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
    except (AttributeError, TypeError, ValueError) as e:
        raise ToolFailure("upstream", f"could not parse the transcript list: {e}") from e

    if not infos:
        return f"No YouTube transcripts found for video {video_id}."

    lines = [f"YouTube transcript languages for {video_id}:"]
    for info in infos:
        kind = "generated" if info.is_generated else "manual"
        translatable = "translatable" if info.is_translatable else "not translatable"
        lines.append(f"- {info.language_code}: {info.language} ({kind}, {translatable})")
        if info.translation_languages:
            sample = ", ".join(
                f"{code} ({language})" for code, language in info.translation_languages[:8]
            )
            extra = len(info.translation_languages) - 8
            if extra > 0:
                sample = f"{sample}, +{extra} more"
            lines.append(f"  translations: {sample}")
    return "\n".join(lines)


@tool(capability="network")
def youtube_transcript_search(
    video_url_or_id: str,
    query: str,
    languages: str = "en",
    prefer_manual: bool = True,
    allow_generated: bool = True,
    max_results: int = 10,
    context_segments: int = 1,
    preserve_formatting: bool = False,
) -> str:
    """Search within a public YouTube transcript and return timestamped matches.

    Args:
        video_url_or_id: YouTube video URL or 11-character video ID.
        query: Case-insensitive text to find in transcript segments.
        languages: Comma-separated preferred source language codes, e.g. "en,pt-BR".
        prefer_manual: Prefer manually-created captions over auto-generated captions.
        allow_generated: Allow auto-generated captions if manual captions are unavailable.
        max_results: Maximum matching windows to return (1-20). Defaults to 10.
        context_segments: Number of neighboring segments around each match (0-3).
        preserve_formatting: Preserve HTML formatting where supported by the provider.

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

    max_results = _clamp(max_results, 1, _MAX_SEARCH_RESULTS)
    context_segments = _clamp(context_segments, 0, 3)
    transcript, segments = _fetch_transcript(
        video_id,
        language_codes,
        prefer_manual=prefer_manual,
        allow_generated=allow_generated,
        translate_to="",
        preserve_formatting=preserve_formatting,
    )

    needle = query.casefold()
    matches = [
        index for index, segment in enumerate(segments) if needle in segment.text.casefold()
    ]
    if not matches:
        return f'No matches found for "{query}" in YouTube transcript {video_id}.'

    lines = [
        f'YouTube transcript matches for "{query}" in {video_id}:',
        _transcript_meta_line(transcript),
    ]
    for result_index, segment_index in enumerate(matches[:max_results], start=1):
        start = max(0, segment_index - context_segments)
        end = min(len(segments), segment_index + context_segments + 1)
        window = segments[start:end]
        start_time = _timestamp(window[0].start, decimal=True)
        end_time = _timestamp(window[-1].end, decimal=True)
        text = " ".join(segment.text.replace("\n", " ").strip() for segment in window)
        lines.append(f"{result_index}. [{start_time} - {end_time}] {text}")

    remaining = len(matches) - max_results
    if remaining > 0:
        lines.append(f"... {remaining} more matches not shown.")
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
) -> tuple[Any, list[_TranscriptSegment]]:
    """The chosen transcript (translated when asked) and its segments.

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
        if translate_to:
            transcript = transcript.translate(translate_to)
        segments = _segments(transcript.fetch(preserve_formatting=preserve_formatting))
    except error_cls as e:
        raise _transcript_failure(e, video_id, languages, translate_to) from e
    except OSError as e:  # requests' network errors
        raise _network_failure(e) from e
    except (AttributeError, TypeError, ValueError) as e:
        raise ToolFailure("upstream", f"could not parse the transcript response: {e}") from e
    return transcript, segments


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


def _transcript_header(video_id: str, transcript: Any) -> str:
    return f"YouTube transcript for {video_id}:\n{_transcript_meta_line(transcript)}"


def _transcript_meta_line(transcript: Any) -> str:
    kind = "generated" if bool(_attr(transcript, "is_generated")) else "manual"
    language = str(_attr(transcript, "language"))
    language_code = str(_attr(transcript, "language_code"))
    return f"Language: {language_code} ({language}) | kind: {kind}"


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


def _limit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    suffix = f"\n... truncated {omitted} characters. Increase max_chars for more transcript text."
    return f"{text[: max(0, max_chars - len(suffix))].rstrip()}{suffix}"


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(int(value), maximum))
