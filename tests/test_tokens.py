"""Tests for _tokens.py — local token counting."""

from __future__ import annotations

import math

from ai_arch_toolkit.core._tokens import (
    _get_correction,
    chars_to_tokens,
    count_tokens_local,
    count_tokens_local_batch,
    tokens_to_chars,
)


class TestGetCorrection:
    def test_openai_exact(self):
        assert _get_correction("gpt-4o") == 1.0
        assert _get_correction("gpt-4o-mini") == 1.0

    def test_o_series(self):
        assert _get_correction("o3-mini") == 1.0
        assert _get_correction("o4-mini") == 1.0

    def test_claude_4x(self):
        assert _get_correction("claude-sonnet-4-6") == 1.15

    def test_claude_3x_longer_prefix_wins(self):
        # "claude-3-" is longer than "claude-" so 1.12 wins
        assert _get_correction("claude-3-5-sonnet-20241022") == 1.12

    def test_gemini(self):
        assert _get_correction("gemini-2.5-pro") == 1.05

    def test_grok(self):
        assert _get_correction("grok-3") == 1.05

    def test_unknown_defaults_to_1(self):
        assert _get_correction("unknown-model") == 1.0


class TestCountTokensLocal:
    def test_basic_count(self):
        n = count_tokens_local("Hello world", model="gpt-4o")
        assert n > 0
        assert isinstance(n, int)

    def test_correction_applied_for_claude(self):
        raw = count_tokens_local("Hello world", model="gpt-4o", correction=1.0)
        corrected = count_tokens_local("Hello world", model="claude-sonnet-4-6")
        # Claude correction is 1.15 so corrected >= raw
        assert corrected >= raw

    def test_correction_override(self):
        raw = count_tokens_local("Hello world", model="claude-sonnet-4-6", correction=1.0)
        doubled = count_tokens_local("Hello world", model="claude-sonnet-4-6", correction=2.0)
        assert doubled == math.ceil(raw * 2.0)

    def test_empty_string(self):
        assert count_tokens_local("", model="gpt-4o") == 0

    def test_o200k_encoding_for_gpt5(self):
        # Should not raise — uses o200k_base
        n = count_tokens_local("Hello world", model="gpt-5")
        assert n > 0


class TestCountTokensLocalBatch:
    def test_batch_equals_sum(self):
        texts = ["Hello world", "This is a test"]
        batch_total = count_tokens_local_batch(texts, model="gpt-4o", correction=1.0)
        individual_sum = sum(count_tokens_local(t, model="gpt-4o", correction=1.0) for t in texts)
        # Due to ceiling, batch may differ slightly but should be close
        assert abs(batch_total - individual_sum) <= len(texts)

    def test_empty_list(self):
        assert count_tokens_local_batch([], model="gpt-4o") == 0


class TestCharsToTokens:
    def test_basic(self):
        # 400 chars / 4 chars_per_token = 100 tokens * 1.0 correction
        assert chars_to_tokens(400, model="gpt-4o") == 100

    def test_with_correction(self):
        # 400 chars / 4 = 100 * 1.15 = 115
        assert chars_to_tokens(400, model="claude-sonnet-4-6") == 115

    def test_zero(self):
        assert chars_to_tokens(0) == 0


class TestTokensToChars:
    def test_basic(self):
        assert tokens_to_chars(100) == 400

    def test_zero(self):
        assert tokens_to_chars(0) == 0

    def test_roundtrip(self):
        chars = tokens_to_chars(100)
        # Using gpt-4o (correction=1.0) for exact roundtrip
        tokens = chars_to_tokens(chars, model="gpt-4o")
        assert tokens == 100


class TestImportGuard:
    def test_importable_from_core(self):
        from ai_arch_toolkit.core import count_tokens_local as fn

        assert callable(fn)

    def test_importable_from_top_level(self):
        from ai_arch_toolkit import count_tokens_local as fn

        assert callable(fn)
