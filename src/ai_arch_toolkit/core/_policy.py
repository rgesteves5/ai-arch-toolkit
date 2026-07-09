"""Policy — execution constraints for Steps."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ai_arch_toolkit.core._retry import RetryConfig

if TYPE_CHECKING:
    from ai_arch_toolkit.core._step import Step

type OnExhausted = Literal["halt", "continue", "fallback"]
type OnLowConfidence = Literal["retry", "escalate", "fallback"]
type OnTimeout = Literal["halt", "fallback"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Policy:
    """Execution policy for a Step — retry, timeout, confidence, cost limits."""

    retry: RetryConfig = field(default_factory=lambda: RetryConfig(max_retries=0))
    timeout: float | None = None
    fallback: Step | None = None
    confidence_threshold: float | None = None
    max_cost: float | None = None
    on_exhausted: OnExhausted = "halt"
    on_low_confidence: OnLowConfidence = "retry"
    on_timeout: OnTimeout = "halt"

    def __post_init__(self) -> None:
        # Reject non-finite caps (NaN/inf): `x <= 0` is False for NaN, so a NaN cap would silently
        # disable enforcement — `anything > NaN` is always False (a money/time fail-open).
        if self.timeout is not None and (not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError(f"timeout must be positive and finite, got {self.timeout}")
        if self.confidence_threshold is not None and not 0 <= self.confidence_threshold <= 1:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )
        if self.max_cost is not None and (not math.isfinite(self.max_cost) or self.max_cost <= 0):
            raise ValueError(f"max_cost must be positive and finite, got {self.max_cost}")
