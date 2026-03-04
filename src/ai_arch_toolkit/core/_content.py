"""Message constructor helpers and multimodal content types."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Multimodal content types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ImagePart:
    """An image content part for multimodal messages.

    ``source`` can be:
    - A URL string (``"https://..."`` or ``"data:..."``).
    - A base64-encoded string.
    - Raw ``bytes``.
    """

    source: str | bytes
    media_type: str = "image/png"


@dataclass(frozen=True, slots=True)
class DocumentPart:
    """A document content part (e.g. PDF) for multimodal messages."""

    source: str | bytes
    media_type: str = "application/pdf"
    name: str | None = None


@dataclass(frozen=True, slots=True)
class CachePart:
    """A text content part annotated for prompt caching (Anthropic)."""

    content: str
    ttl: str = "ephemeral"


type ContentPart = str | ImagePart | DocumentPart | CachePart
type Content = str | list[ContentPart]


# ---------------------------------------------------------------------------
# Content helper constructors
# ---------------------------------------------------------------------------


def image(source: str | bytes, media_type: str = "image/png") -> ImagePart:
    """Create an image content part."""
    return ImagePart(source=source, media_type=media_type)


def document(
    source: str | bytes,
    media_type: str = "application/pdf",
    name: str | None = None,
) -> DocumentPart:
    """Create a document content part."""
    return DocumentPart(source=source, media_type=media_type, name=name)


def cache(content: str) -> CachePart:
    """Create a cache-control annotated text part (Anthropic prompt caching)."""
    return CachePart(content=content)


def _encode_b64(source: str | bytes) -> str:
    """Ensure source is a base64 string. Encodes raw bytes if needed."""
    if isinstance(source, bytes):
        return base64.b64encode(source).decode("ascii")
    return source


def _is_url(source: str | bytes) -> bool:
    """Check if source looks like a URL."""
    if isinstance(source, bytes):
        return False
    return source.startswith(("https://", "http://", "data:"))


# ---------------------------------------------------------------------------
# Message constructors
# ---------------------------------------------------------------------------


def system(content: str) -> dict[str, Any]:
    """Create a system message dict."""
    return {"role": "system", "content": content}


def user(content: Content) -> dict[str, Any]:
    """Create a user message dict.

    Accepts a plain string or a list of content parts (text, images, documents).
    """
    return {"role": "user", "content": content}


def assistant(content: str) -> dict[str, Any]:
    """Create an assistant message dict."""
    return {"role": "assistant", "content": content}


def tool_result(
    content: Any,
    *,
    tool_use_id: str,
    name: str | None = None,
) -> dict[str, Any]:
    """Create a tool_result message dict.

    ``tool_use_id`` is the provider-agnostic discriminator for tool results.
    """
    if not tool_use_id:
        raise ValueError("tool_use_id must be a non-empty string")
    msg = {"role": "tool", "content": content, "tool_use_id": tool_use_id}
    if name:
        msg["name"] = name
    return msg
