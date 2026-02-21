"""Tests for token counting helpers and model correction factors."""

from __future__ import annotations

from ai_arch_toolkit.llm._tokens import (
    CLAUDE_3_CORRECTION_FACTOR,
    CLAUDE_4_CORRECTION_FACTOR,
    DEFAULT_CORRECTION_FACTOR,
    GEMINI_CORRECTION_FACTOR,
    GROK_CORRECTION_FACTOR,
    META_CORRECTION_FACTOR,
    TokenCorrectionConfig,
    estimate_content_tokens,
    estimate_conversation_tokens,
    estimate_conversation_tokens_for_model,
    estimate_item_tokens,
    estimate_message_tokens,
    estimate_text_tokens,
    estimate_text_tokens_for_model,
    get_correction_factor,
)
from ai_arch_toolkit.llm._types import Message, TextPart, ToolCall, ToolResult


def test_estimate_text_tokens_len_div_4() -> None:
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("abcdefgh") == 2


def test_estimate_message_tokens_includes_overhead_and_tool_calls() -> None:
    message = Message(
        role="assistant",
        content="hello world",
        tool_calls=(ToolCall(id="1", name="search", arguments={"q": "test"}),),
    )
    assert estimate_message_tokens(message) >= 4


def test_estimate_content_tokens_multimodal() -> None:
    content = (TextPart(text="hello"),)
    assert estimate_content_tokens(content) > 0


def test_estimate_item_tokens_for_tool_result() -> None:
    item = ToolResult(tool_call_id="c1", name="search", content="result")
    assert estimate_item_tokens(item) > 0


def test_estimate_conversation_tokens_sums_items() -> None:
    items = [Message(role="user", content="hello"), Message(role="assistant", content="world")]
    total = estimate_conversation_tokens(items)
    assert total == estimate_item_tokens(items[0]) + estimate_item_tokens(items[1])


def test_get_correction_factor_by_provider_family() -> None:
    assert get_correction_factor("claude-3-5-sonnet-latest") == CLAUDE_3_CORRECTION_FACTOR
    assert get_correction_factor("claude-sonnet-4-5") == CLAUDE_4_CORRECTION_FACTOR
    assert get_correction_factor("gemini-2.0-flash") == GEMINI_CORRECTION_FACTOR
    assert get_correction_factor("xai/grok-4-0709") == GROK_CORRECTION_FACTOR
    assert get_correction_factor("meta-llama-4-maverick") == META_CORRECTION_FACTOR
    assert get_correction_factor("gpt-4o") == DEFAULT_CORRECTION_FACTOR


def test_get_correction_factor_uses_exact_model_override_case_insensitive() -> None:
    config = TokenCorrectionConfig(model_overrides={"GPT-4O-MINI": 1.37})
    assert get_correction_factor("gpt-4o-mini", config=config) == 1.37


def test_estimate_text_tokens_for_model_applies_requested_formula() -> None:
    tokens = estimate_text_tokens_for_model(
        "hello",
        "claude-3-5-sonnet-latest",
        raw_token_counter=lambda _text, _model: 100,
    )
    assert tokens == int(100 * CLAUDE_3_CORRECTION_FACTOR)


def test_estimate_conversation_tokens_for_model_with_custom_override() -> None:
    items = [Message(role="user", content="hello")]
    config = TokenCorrectionConfig(model_overrides={"my-model": 1.5})
    total = estimate_conversation_tokens_for_model(
        items,
        "my-model",
        correction_config=config,
        raw_token_counter=lambda _text, _model: 10,
    )
    assert total == 4 + int(10 * 1.5)
