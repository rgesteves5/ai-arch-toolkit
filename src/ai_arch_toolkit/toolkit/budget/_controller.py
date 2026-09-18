"""Pure admission policy; the store repeats the neutral limit check atomically."""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_arch_toolkit.core._metering._admission import (
    AdmissionDecision,
    MeterSnapshot,
    Reservation,
    limit_denial,
)
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._operation import OperationRequest
from ai_arch_toolkit.toolkit.budget._estimator import Estimator, HeuristicEstimator
from ai_arch_toolkit.toolkit.budget._exceptions import BudgetExceeded
from ai_arch_toolkit.toolkit.budget._policy import BudgetPolicy

__all__ = ["BudgetController"]


@dataclass(frozen=True, slots=True)
class BudgetController:
    """Admit operations and bound failures using the same estimator."""

    policy: BudgetPolicy
    estimator: Estimator = field(default_factory=HeuristicEstimator)

    def wants_request_size(self) -> bool:
        """Only strict admission needs the size; soft failures size lazily."""
        return self.policy.reserve == "strict"

    def failure_bound(self, request: OperationRequest, reservation: Reservation) -> Money | None:
        """Retain the operation's strict hold, or estimate its bound after a soft failure."""
        if self.policy.reserve == "strict":
            return reservation.cost
        estimate = self.estimator.estimate(request)
        return estimate.cost if estimate is not None else None

    def admit(self, snapshot: MeterSnapshot, request: OperationRequest) -> AdmissionDecision:
        reservation = Reservation()
        if self.policy.reserve == "strict":
            estimate = self.estimator.estimate(request)
            if estimate is None:
                return AdmissionDecision.deny(
                    BudgetExceeded(
                        "cannot price this operation under a strict budget",
                        dimension="cost",
                    )
                )
            reservation = estimate
        denial = self._exceeds(snapshot, request, reservation)
        if denial is not None:
            return AdmissionDecision.deny(denial)
        return AdmissionDecision.allow(reservation, self.policy.to_limits())

    def _exceeds(
        self,
        snap: MeterSnapshot,
        request: OperationRequest,
        reservation: Reservation,
    ) -> BudgetExceeded | None:
        denial = limit_denial(snap, self.policy.to_limits(), request, reservation)
        if denial is not None:
            return BudgetExceeded(
                dimension=denial.dimension,
                limit=denial.limit,
                current=denial.current,
                attempted=denial.attempted,
            )
        if (
            self.policy.max_cost is not None
            and self.policy.unpriced == "fail_closed"
            and snap.unknown_cost_count > 0
        ):
            return BudgetExceeded(
                "a prior call could not be priced — failing closed under a cost cap",
                dimension="cost",
                limit=self.policy.max_cost,
            )
        return None
