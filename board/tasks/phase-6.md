# Phase 6: Multimodal Content

**Status**: Queued (after Phase 5)
**Why**: Vision and document agents need images/PDFs in messages.

## New Types (`core/_content.py` — input types)

These are input types (go in messages), not output types. They live in `_content.py`.

- `TextPart(text: str)` — frozen dataclass
- `ImagePart(url: str = "", media_type: str = "", data: str = "", detail: str = "auto")` — frozen
- `AudioPart(data: str = "", media_type: str = "", transcript: str = "")` — frozen
- `DocumentPart(data: str = "", media_type: str = "application/pdf", uri: str = "")` — frozen
- `type ContentPart = TextPart | ImagePart | AudioPart | DocumentPart`
- `type Content = str | tuple[ContentPart, ...]`

## Content Builders (`core/_content.py`)

- `image_block(url=None, data=None, media_type=None, detail="auto") -> ImagePart`
- `audio_block(data, media_type, transcript="") -> AudioPart`
- `document_block(data=None, media_type="application/pdf", uri=None) -> DocumentPart`
- Update `user()` signature to accept `str | list[str | ContentPart]`

## Provider Wire Format

- **Anthropic** `_content_to_anthropic()`: ImagePart → base64/URL source, DocumentPart → base64, AudioPart → ValueError
- **OpenAI** `_content_to_openai()`: ImagePart → image_url, AudioPart → input_audio, DocumentPart → ValueError
- Both: `_messages_to_wire` detects non-string content via `isinstance(content, (list, tuple))`

## Tests: `tests/test_multimodal.py`

- Each part type through each provider's wire converter
- Error cases (AudioPart → Anthropic, DocumentPart → OpenAI)
- Round-trip: user(ImagePart) → _messages_to_wire → correct format
