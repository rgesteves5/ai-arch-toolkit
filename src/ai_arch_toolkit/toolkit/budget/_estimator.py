"""Pre-call estimation — turns an operation's facts into a worst-case token/cost reservation.

Consulted for strict admission. The worst case itself is core's (``core/_metering/_worst_case.py``,
D49), which also bounds every failure the meter cannot settle; an :class:`Estimator` is where a
budget swaps in another opinion of what an operation may cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ai_arch_toolkit.core._metering._admission import Reservation
from ai_arch_toolkit.core._metering._operation import OperationRequest
from ai_arch_toolkit.core._metering._scope import Pricer
from ai_arch_toolkit.core._metering._worst_case import worst_case
from ai_arch_toolkit.core._pricing import pricing

__all__ = ["Estimator", "HeuristicEstimator"]


class Estimator(Protocol):
    """Estimates a per-operation :class:`Reservation`, or ``None`` when it cannot price it."""

    def estimate(self, request: OperationRequest) -> Reservation | None: ...


@dataclass(frozen=True, slots=True)
class HeuristicEstimator:
    """The worst case of an operation's facts (core's ``worst_case``), at ``pricer``'s prices.

    Returns ``None`` when the model is unpriced — the signal for a strict controller to fail
    closed (deny) rather than admit an uncosted call. Tools reserve their Pricer cost.
    """

    pricer: Pricer | None = None

    def estimate(self, request: OperationRequest) -> Reservation | None:
        return worst_case(request, self.pricer or pricing)
