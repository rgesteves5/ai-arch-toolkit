"""Pre-call estimation — turns an operation's facts into a worst-case token/cost reservation.

Only consulted under ``reserve="strict"``. Opinion (the char/token ratio, the pricing source)
lives here in ``toolkit``, never in the neutral core meter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ai_arch_toolkit.core._metering._admission import Reservation
from ai_arch_toolkit.core._metering._operation import OperationRequest
from ai_arch_toolkit.core._metering._scope import Pricer
from ai_arch_toolkit.core._pricing import pricing
from ai_arch_toolkit.core._response import Usage

__all__ = ["Estimator", "HeuristicEstimator"]

_CHARS_PER_TOKEN = 4  # rough English-text ratio; deliberately conservative


class Estimator(Protocol):
    """Estimates a per-operation :class:`Reservation`, or ``None`` when it cannot price it."""

    def estimate(self, request: OperationRequest) -> Reservation | None: ...


@dataclass(frozen=True, slots=True)
class HeuristicEstimator:
    """Worst-case reservation from ``content_size_hint`` + ``declared_max_output_tokens``.

    Returns ``None`` when the model is unpriced — the signal for a strict controller to fail
    closed (deny) rather than admit an uncosted call. Non-LLM ops reserve nothing.
    """

    pricer: Pricer | None = None

    def estimate(self, request: OperationRequest) -> Reservation | None:
        if request.kind != "llm":
            return Reservation()
        input_tokens = (request.content_size_hint or 0) // _CHARS_PER_TOKEN
        output_tokens = request.declared_max_output_tokens or 0
        usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
        cost = (self.pricer or pricing).price(request, usage)
        if cost.kind == "unknown" or cost.amount is None:
            return None  # unpriced under a strict budget -> caller fails closed
        return Reservation(
            input_tokens=input_tokens, output_tokens=output_tokens, cost=cost.amount
        )
