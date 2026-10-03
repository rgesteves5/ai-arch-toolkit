"""Tests for central redaction utilities."""

from __future__ import annotations

from ai_arch_toolkit.core._redaction import RedactionPolicy, Redactor, redact_text


def test_redact_text_masks_common_secret_shapes() -> None:
    text = """
OPENAI_API_KEY=sk-testsecret1234567890
Authorization: Bearer abc.def.ghi
DATABASE_URL=postgresql://user:pass@example.com/db
-----BEGIN PRIVATE KEY-----
abc123
-----END PRIVATE KEY-----
"""

    redacted = redact_text(text)

    assert "sk-testsecret1234567890" not in redacted
    assert "abc.def.ghi" not in redacted
    assert "user:pass@example.com" not in redacted
    assert "BEGIN PRIVATE KEY" not in redacted
    assert "[REDACTED]" in redacted


def test_sensitive_dict_keys_are_replaced() -> None:
    redactor = Redactor()

    payload = redactor.redact(
        {
            "api_key": "sk-testsecret1234567890",
            "nested": {"password": "secret-password"},
            "safe": "visible",
        }
    )

    assert payload["api_key"] == "[REDACTED]"
    assert payload["nested"]["password"] == "[REDACTED]"
    assert payload["safe"] == "visible"


def test_full_debug_policy_returns_unredacted_text() -> None:
    redactor = Redactor(RedactionPolicy(trace_mode="full_debug"))

    text = "OPENAI_API_KEY=sk-testsecret1234567890"

    assert redactor.redact_text(text) == text


def test_provider_keys_are_masked_by_their_prefix() -> None:
    xai = "xai-" + "A1b2C3d4" * 10
    groq = "gsk_" + "Zz9Yy8Xx" * 6 + "Ww7v"
    google = "AIza" + "SyD-x_9" * 5  # 39 characters; may hold "-" and "_"
    google_dash_end = "AIza" + "B" * 34 + "-"
    text = f"xai {xai}, groq {groq}, google {google} and {google_dash_end}."

    redacted = redact_text(text)

    for key in (xai, groq, google, google_dash_end):
        assert key not in redacted
    assert redacted == "xai [REDACTED], groq [REDACTED], google [REDACTED] and [REDACTED]."


def test_words_that_only_look_like_key_prefixes_stay() -> None:
    text = "taxai-like words, gsk_short, AIzaTooShort and a maxai-1234567890abcdefghij word"
    assert redact_text(text) == text
