"""Tests for _content.py message constructors and multimodal types."""

from __future__ import annotations

import base64

import pytest

from ai_arch_toolkit.core._content import (
    CachePart,
    DocumentPart,
    ImagePart,
    _encode_b64,
    _is_url,
    assistant,
    cache,
    document,
    image,
    system,
    tool_result,
    user,
)


class TestSystem:
    def test_returns_dict_with_role_and_content(self):
        msg = system("You are helpful.")
        assert msg == {"role": "system", "content": "You are helpful."}


class TestUser:
    def test_string_content(self):
        msg = user("Hello")
        assert msg == {"role": "user", "content": "Hello"}

    def test_list_content_for_multimodal(self):
        parts = [{"type": "text", "text": "Hi"}, {"type": "image_url", "url": "http://x"}]
        msg = user(parts)
        assert msg == {"role": "user", "content": parts}


class TestAssistant:
    def test_returns_dict_with_role_and_content(self):
        msg = assistant("Sure!")
        assert msg == {"role": "assistant", "content": "Sure!"}


class TestToolResult:
    def test_includes_tool_use_id(self):
        msg = tool_result("result data", tool_use_id="call_123")
        assert msg["role"] == "tool"
        assert msg["content"] == "result data"
        assert msg["tool_use_id"] == "call_123"

    def test_includes_optional_name(self):
        msg = tool_result("result data", tool_use_id="call_123", name="get_weather")
        assert msg["name"] == "get_weather"


class TestToolResultValidation:
    def test_empty_tool_use_id_raises(self):
        with pytest.raises(ValueError, match="tool_use_id must be a non-empty string"):
            tool_result("result", tool_use_id="")


# ---------------------------------------------------------------------------
# Multimodal types
# ---------------------------------------------------------------------------


class TestImagePart:
    def test_defaults(self):
        part = ImagePart(source="abc123")
        assert part.source == "abc123"
        assert part.media_type == "image/png"

    def test_custom_media_type(self):
        part = ImagePart(source=b"\x89PNG", media_type="image/jpeg")
        assert part.media_type == "image/jpeg"

    def test_frozen(self):
        part = ImagePart(source="x")
        with pytest.raises(AttributeError):
            part.source = "y"  # type: ignore[misc]

    def test_image_helper(self):
        part = image("https://example.com/img.png", media_type="image/webp")
        assert isinstance(part, ImagePart)
        assert part.media_type == "image/webp"


class TestDocumentPart:
    def test_defaults(self):
        part = DocumentPart(source="pdf_data")
        assert part.media_type == "application/pdf"
        assert part.name is None

    def test_with_name(self):
        part = DocumentPart(source=b"\x25PDF", name="report.pdf")
        assert part.name == "report.pdf"

    def test_document_helper(self):
        part = document(b"\x25PDF", name="report.pdf")
        assert isinstance(part, DocumentPart)
        assert part.name == "report.pdf"


class TestCachePart:
    def test_defaults(self):
        part = CachePart(content="long context")
        assert part.content == "long context"
        assert part.ttl == "ephemeral"

    def test_cache_helper(self):
        part = cache("context text")
        assert isinstance(part, CachePart)
        assert part.content == "context text"


class TestEncodeb64:
    def test_bytes(self):
        raw = b"hello"
        assert _encode_b64(raw) == base64.b64encode(raw).decode("ascii")

    def test_string_passthrough(self):
        assert _encode_b64("already_b64") == "already_b64"


class TestIsUrl:
    def test_https(self):
        assert _is_url("https://example.com/img.png") is True

    def test_b64_string(self):
        assert _is_url("abc123") is False

    def test_bytes(self):
        assert _is_url(b"\x89PNG") is False


class TestUserWithMultimodal:
    def test_image_part_in_list(self):
        img = image("https://example.com/img.png")
        msg = user(["Describe this:", img])
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert isinstance(msg["content"][1], ImagePart)
