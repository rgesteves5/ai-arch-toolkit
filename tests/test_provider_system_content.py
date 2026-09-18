"""System messages take text parts, and cache() parts reach every adapter as their text."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from ai_arch_toolkit.core._content import cache, document, image, system, user
from ai_arch_toolkit.core._providers import _anthropic, _gemini, _openai, _xai
from tests.provider_calls import complete

SYSTEM_WITH_PARTS = {"role": "system", "content": ["Be terse.", cache("POLICY")]}
SYSTEM_WITH_IMAGE = {"role": "system", "content": ["Look:", image("https://example.com/a.png")]}
USER_WITH_CACHE = user(["Answer briefly.", cache("LONG CONTEXT"), "What is a decorator?"])
CACHED_POLICY = {"type": "text", "text": "POLICY", "cache_control": {"type": "ephemeral"}}


class TestOpenAI:
    def test_system_parts_are_sent_as_text(self) -> None:
        wire = _openai._messages_to_sdk([SYSTEM_WITH_PARTS, user("hi")], system="Rules.")

        assert wire[:2] == [
            {"role": "system", "content": "Rules."},
            {"role": "system", "content": "Be terse.\n\nPOLICY"},
        ]

    def test_cache_parts_in_user_content_are_sent_as_their_text(self) -> None:
        wire = _openai._messages_to_sdk([USER_WITH_CACHE])

        texts = [block["text"] for block in wire[0]["content"]]
        assert texts == ["Answer briefly.", "LONG CONTEXT", "What is a decorator?"]

    def test_an_image_in_a_system_message_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="user message"):
            _openai._messages_to_sdk([SYSTEM_WITH_IMAGE])


class TestGemini:
    def test_system_parts_are_joined_as_text(self) -> None:
        system_text, _ = _gemini._messages_to_sdk([SYSTEM_WITH_PARTS, user("hi")])

        assert system_text == "Be terse.\n\nPOLICY"

    def test_cache_parts_in_user_content_are_sent_as_their_text(self) -> None:
        _, contents = _gemini._messages_to_sdk([USER_WITH_CACHE])

        texts = [part.text for part in contents[0].parts]
        assert texts == ["Answer briefly.", "LONG CONTEXT", "What is a decorator?"]

    def test_an_image_in_a_system_message_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="user message"):
            _gemini._messages_to_sdk([SYSTEM_WITH_IMAGE])


class TestXai:
    def test_system_parts_are_joined_as_text(self) -> None:
        _, system_text = _xai._messages_to_sdk([SYSTEM_WITH_PARTS, user("hi")])

        assert system_text == "Be terse.\n\nPOLICY"

    def test_cache_parts_in_user_content_are_sent_as_their_text(self) -> None:
        messages, _ = _xai._messages_to_sdk([USER_WITH_CACHE])

        assert messages[0].content[0].text == "Answer briefly.\nLONG CONTEXT\nWhat is a decorator?"

    def test_a_document_is_dropped_with_a_warning_instead_of_sent_as_text(self) -> None:
        with pytest.warns(UserWarning, match="document"):
            messages, _ = _xai._messages_to_sdk([user(["Summarise:", document(b"%PDF-1.4")])])

        assert messages[0].content[0].text == "Summarise:"

    def test_an_image_in_a_system_message_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="user message"):
            _xai._messages_to_sdk([SYSTEM_WITH_IMAGE])


def _anthropic_message() -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        output_tokens_details=None,
    )
    text = SimpleNamespace(type="text", text="{}", citations=None)
    return SimpleNamespace(
        id="msg_test",
        content=[text],
        model="claude-sonnet-4-6",
        stop_reason="end_turn",
        usage=usage,
    )


class TestAnthropic:
    def test_text_system_parts_are_joined_into_one_string(self) -> None:
        messages = [system("A"), {"role": "system", "content": ["B", "C"]}, user("hi")]

        system_param, _ = _anthropic._messages_to_sdk(messages)

        assert system_param == "A\n\nB\n\nC"

    def test_a_cached_system_part_keeps_its_cache_marker(self) -> None:
        system_param, _ = _anthropic._messages_to_sdk([SYSTEM_WITH_PARTS, user("hi")])

        assert system_param == [{"type": "text", "text": "Be terse."}, CACHED_POLICY]

    def test_an_image_in_a_system_message_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="user message"):
            _anthropic._messages_to_sdk([SYSTEM_WITH_IMAGE])

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_complete_keeps_the_cache_marker_after_merging_and_json_mode(
        self, mock_sdk
    ) -> None:
        client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = client
        client.messages.create.return_value = _anthropic_message()
        provider = _anthropic.AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = client

        await complete(
            provider,
            [SYSTEM_WITH_PARTS, user("hi")],
            system="Rules.",
            json_mode=True,
            max_tokens=1024,
        )

        assert client.messages.create.call_args.kwargs["system"] == [
            {"type": "text", "text": "Rules."},
            {"type": "text", "text": "Be terse."},
            CACHED_POLICY,
            {"type": "text", "text": "Respond with valid JSON only."},
        ]

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_count_tokens_sends_the_cached_blocks(self, mock_sdk) -> None:
        client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = client
        client.messages.count_tokens.return_value = SimpleNamespace(input_tokens=42)
        provider = _anthropic.AnthropicProvider("claude-sonnet-4-6", "test-key")
        provider._client = client

        tokens = await provider.count_tokens([SYSTEM_WITH_PARTS, user("hi")])

        assert tokens == 42
        assert client.messages.count_tokens.call_args.kwargs["system"] == [
            {"type": "text", "text": "Be terse."},
            CACHED_POLICY,
        ]
