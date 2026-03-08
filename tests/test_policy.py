"""Tests for Policy validation."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._retry import RetryConfig


class TestPolicy:
    def test_defaults(self) -> None:
        p = Policy()
        assert p.retry.max_retries == 0
        assert p.timeout is None
        assert p.confidence_threshold is None
        assert p.max_cost is None
        assert p.on_exhausted == "halt"
        assert p.on_low_confidence == "retry"
        assert p.on_timeout == "halt"

    def test_custom_retry(self) -> None:
        p = Policy(retry=RetryConfig(max_retries=3, base_delay=0.5))
        assert p.retry.max_retries == 3
        assert p.retry.base_delay == 0.5

    def test_timeout_validation(self) -> None:
        with pytest.raises(ValueError, match="timeout must be positive"):
            Policy(timeout=0)
        with pytest.raises(ValueError, match="timeout must be positive"):
            Policy(timeout=-1)

    def test_confidence_validation(self) -> None:
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            Policy(confidence_threshold=1.5)
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            Policy(confidence_threshold=-0.1)
        # Valid edges
        Policy(confidence_threshold=0.0)
        Policy(confidence_threshold=1.0)

    def test_max_cost_validation(self) -> None:
        with pytest.raises(ValueError, match="max_cost must be positive"):
            Policy(max_cost=0)
        with pytest.raises(ValueError, match="max_cost must be positive"):
            Policy(max_cost=-5)

    def test_valid_full_policy(self) -> None:
        p = Policy(
            retry=RetryConfig(max_retries=2),
            timeout=30.0,
            confidence_threshold=0.7,
            max_cost=1.0,
            on_exhausted="fallback",
            on_low_confidence="escalate",
            on_timeout="fallback",
        )
        assert p.timeout == 30.0
        assert p.confidence_threshold == 0.7
        assert p.max_cost == 1.0
