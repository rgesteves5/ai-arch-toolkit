"""Pre-call estimation — turns an operation's facts into a worst-case token/cost reservation.

Consulted for strict admission and for bounding indeterminate failures under soft budgets.
Opinion (the char/token ratio, the pricing source) lives here in ``toolkit``, never in core.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from ai_arch_toolkit.core._metering._admission import Reservation
from ai_arch_toolkit.core._metering._operation import OperationRequest
from ai_arch_toolkit.core._metering._scope import Pricer
from ai_arch_toolkit.core._pricing import pricing
from ai_arch_toolkit.core._response import Usage

__all__ = ["Estimator", "HeuristicEstimator"]

_CHARS_PER_TOKEN = 4  # rough English-text ratio; deliberately conservative
_NON_TEXT_TOKEN_ALLOWANCE = 4000  # worst-case tokens to reserve per image/document part
# Worst-case image output tokens per generated image. A 4K image is 2,520 tokens on Gemini 3.1
# Flash Image, a high-quality 1536x1024 one 6,240 on gpt-image-1; the largest GPT Image sizes
# and qualities have no published count, hence the margin.
_IMAGE_OUTPUT_TOKEN_ALLOWANCE = 16_000


class Estimator(Protocol):
    """Estimates a per-operation :class:`Reservation`, or ``None`` when it cannot price it."""

    def estimate(self, request: OperationRequest) -> Reservation | None: ...


@dataclass(frozen=True, slots=True)
class HeuristicEstimator:
    """Worst-case reservation from ``content_size_hint`` + ``declared_max_output_tokens``.

    An image generation also reserves, for each declared image, an allowance of image output
    tokens and the model's per-image price.

    Returns ``None`` when the model is unpriced — the signal for a strict controller to fail
    closed (deny) rather than admit an uncosted call. Tools reserve their Pricer cost.
    """

    pricer: Pricer | None = None

    def estimate(self, request: OperationRequest) -> Reservation | None:
        if request.kind != "llm":
            return self._priced(request, Usage(), 0, 0)
        # Round UP (a reservation is a worst-case hold) and add a per-image/document allowance —
        # multimodal parts carry far more tokens than their textual placeholder in the char hint.
        input_tokens = math.ceil((request.content_size_hint or 0) / _CHARS_PER_TOKEN)
        input_tokens += request.non_text_parts * _NON_TEXT_TOKEN_ALLOWANCE
        output_tokens = request.declared_max_output_tokens or 0
        image_tokens = request.declared_images * _IMAGE_OUTPUT_TOKEN_ALLOWANCE
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            image_output_tokens=image_tokens,
            image_count=request.declared_images,
        )
        return self._priced(request, usage, input_tokens, output_tokens + image_tokens)

    def _priced(
        self,
        request: OperationRequest,
        usage: Usage,
        input_tokens: int,
        output_tokens: int,
    ) -> Reservation | None:
        try:
            cost = (self.pricer or pricing).price(request, usage)
        except Exception:
            return None
        if cost.kind == "unknown" or cost.amount is None:
            return None
        return Reservation(
            input_tokens=input_tokens, output_tokens=output_tokens, cost=cost.amount
        )
