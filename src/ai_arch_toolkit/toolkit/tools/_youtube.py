"""YouTube transcript tools powered by youtube-transcript-api."""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core import tool

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


@tool
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
    """
    video_id = _normalize_video_id(video_url_or_id)
    if not video_id:
        return f"YouTube transcript failed: invalid video URL or ID: {video_url_or_id!r}"

    language_codes = _parse_languages(languages)
    if not language_codes:
        return "YouTube transcript failed: provide at least one language code."

    output_format = output_format.strip().lower()
    if output_format not in {"text", "segments", "json", "srt", "vtt"}:
        return (
            'YouTube transcript failed: output_format must be "text", "segments", '
            '"json", "srt", or "vtt".'
        )

    max_chars = _clamp(max_chars, 1, _MAX_CHARS_LIMIT)
    api_cls, error_cls = _load_youtube_transcript_api()
    if api_cls is None:
        return f"YouTube transcript failed: {_OPTIONAL_DEP_ERROR}"

    try:
        transcript = _select_transcript(
            api_cls().list(video_id),
            language_codes,
            prefer_manual=prefer_manual,
            allow_generated=allow_generated,
        )
        if translate_to.strip():
            transcript = transcript.translate(translate_to.strip())
        segments = _segments(transcript.fetch(preserve_formatting=preserve_formatting))
    except error_cls as e:
        return f"YouTube transcript failed: {e}"
    except (AttributeError, TypeError, ValueError) as e:
        return f"YouTube transcript failed: could not parse transcript response: {e}"

    if not segments:
        return f"YouTube transcript failed: no transcript text found for video {video_id}."

    header = _transcript_header(video_id, transcript)
    body = _format_segments(segments, output_format)
    return _limit_text(f"{header}\n{body}", max_chars)


@tool
def youtube_transcript_languages(video_url_or_id: str) -> str:
    """List public transcript languages available for a YouTube video.

    Args:
        video_url_or_id: YouTube video URL or 11-character video ID.
    """
    video_id = _normalize_video_id(video_url_or_id)
    if not video_id:
        return f"YouTube transcript languages failed: invalid video URL or ID: {video_url_or_id!r}"

    api_cls, error_cls = _load_youtube_transcript_api()
    if api_cls is None:
        return f"YouTube transcript languages failed: {_OPTIONAL_DEP_ERROR}"

    try:
        infos = [_transcript_info(transcript) for transcript in api_cls().list(video_id)]
    except error_cls as e:
        return f"YouTube transcript languages failed: {e}"
    except (AttributeError, TypeError, ValueError) as e:
        return f"YouTube transcript languages failed: could not parse transcript list: {e}"

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


@tool
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
    """
    video_id = _normalize_video_id(video_url_or_id)
    if not video_id:
        return f"YouTube transcript search failed: invalid video URL or ID: {video_url_or_id!r}"
    if not query.strip():
        return "YouTube transcript search failed: query must not be empty."

    language_codes = _parse_languages(languages)
    if not language_codes:
        return "YouTube transcript search failed: provide at least one language code."

    max_results = _clamp(max_results, 1, _MAX_SEARCH_RESULTS)
    context_segments = _clamp(context_segments, 0, 3)
    api_cls, error_cls = _load_youtube_transcript_api()
    if api_cls is None:
        return f"YouTube transcript search failed: {_OPTIONAL_DEP_ERROR}"

    try:
        transcript = _select_transcript(
            api_cls().list(video_id),
            language_codes,
            prefer_manual=prefer_manual,
            allow_generated=allow_generated,
        )
        segments = _segments(transcript.fetch(preserve_formatting=preserve_formatting))
    except error_cls as e:
        return f"YouTube transcript search failed: {e}"
    except (AttributeError, TypeError, ValueError) as e:
        return f"YouTube transcript search failed: could not parse transcript response: {e}"

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
