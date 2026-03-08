"""Unit tests for input validation across LLM and RetryConfig."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._retry import RetryConfig

# ---------------------------------------------------------------------------
# LLM.__init__() validation
# ---------------------------------------------------------------------------


class TestLLMInitValidation:
    """Validate constructor guards on temperature, max_tokens, and timeout."""

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_temperature_below_range(self, mock_cp):
        with pytest.raises(ValueError, match=r"temperature must be between 0\.0 and 2\.0"):
            LLM("claude-3-haiku", temperature=-0.1)

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_temperature_above_range(self, mock_cp):
        with pytest.raises(ValueError, match=r"temperature must be between 0\.0 and 2\.0"):
            LLM("claude-3-haiku", temperature=2.1)

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_max_tokens_zero(self, mock_cp):
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            LLM("claude-3-haiku", max_tokens=0)

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_max_tokens_negative(self, mock_cp):
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            LLM("claude-3-haiku", max_tokens=-5)

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_timeout_zero(self, mock_cp):
        with pytest.raises(ValueError, match="timeout must be positive"):
            LLM("claude-3-haiku", timeout=0)

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_timeout_negative(self, mock_cp):
        with pytest.raises(ValueError, match="timeout must be positive"):
            LLM("claude-3-haiku", timeout=-1.0)

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_valid_params_no_error(self, mock_cp):
        llm = LLM("claude-3-haiku", temperature=1.0, max_tokens=100, timeout=30.0)
        assert llm._model == "claude-3-haiku"


# ---------------------------------------------------------------------------
# LLM._normalize() validation
# ---------------------------------------------------------------------------


class TestLLMNormalizeValidation:
    """Validate _normalize() type/value checks on messages input."""

    def test_non_list_non_string_raises_type_error(self):
        with pytest.raises(TypeError, match="messages must be a string or list of dicts"):
            LLM._normalize(123)

    def test_list_with_non_dict_raises_type_error(self):
        with pytest.raises(TypeError, match=r"messages\[0\] must be a dict"):
            LLM._normalize(["not a dict"])

    def test_dict_missing_role_raises_value_error(self):
        with pytest.raises(ValueError, match=r"messages\[0\] missing required 'role' key"):
            LLM._normalize([{"content": "hello"}])


# ---------------------------------------------------------------------------
# RetryConfig validation
# ---------------------------------------------------------------------------


class TestRetryConfigValidation:
    """Validate RetryConfig __post_init__ guards."""

    def test_max_retries_negative(self):
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            RetryConfig(max_retries=-1)

    def test_base_delay_zero(self):
        with pytest.raises(ValueError, match="base_delay must be positive"):
            RetryConfig(base_delay=0)

    def test_max_delay_negative(self):
        with pytest.raises(ValueError, match="max_delay must be positive"):
            RetryConfig(max_delay=-1.0)
