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
