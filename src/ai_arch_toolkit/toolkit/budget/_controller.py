"""``BudgetController`` — the toolkit's :class:`AdmissionController` over a :class:`BudgetPolicy`.

Pure and sync (the meter runs it outside its lock). It pre-checks the snapshot and denies with a
:class:`BudgetExceeded`; the store independently re-validates the returned ``ResourceLimits`` under
its lock, so a stale admit can never overshoot a hard cap (that race raises the neutral base).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_arch_toolkit.core._metering._admission import (
    AdmissionDecision,
    MeterSnapshot,
    Reservation,
)
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._operation import OperationRequest
from ai_arch_toolkit.toolkit.budget._estimator import Estimator, HeuristicEstimator
from ai_arch_toolkit.toolkit.budget._exceptions import BudgetExceeded
from ai_arch_toolkit.toolkit.budget._policy import BudgetPolicy

__all__ = ["BudgetController"]


@dataclass(frozen=True, slots=True)
class BudgetController:
    """Admits operations against a :class:`BudgetPolicy`."""

    policy: BudgetPolicy
    estimator: Estimator = field(default_factory=HeuristicEstimator)

    def admit(self, snapshot: MeterSnapshot, request: OperationRequest) -> AdmissionDecision:
        reservation = Reservation()
        if self.policy.reserve == "strict":
            estimate = self.estimator.estimate(request)
            if estimate is None:
                return AdmissionDecision.deny(
                    BudgetExceeded(
                        "cannot price this operation under a strict budget", dimension="cost"
                    )
                )
            reservation = estimate
        denial = self._exceeds(snapshot, request, reservation)
        if denial is not None:
            return AdmissionDecision.deny(denial)
        return AdmissionDecision.allow(reservation, self.policy.to_limits())

    def _exceeds(
        self, snap: MeterSnapshot, request: OperationRequest, reservation: Reservation
    ) -> BudgetExceeded | None:
        """Prospective committed + outstanding + this op vs each cap (mirrors the store)."""
        p = self.policy
        add_llm = request.count if request.kind == "llm" else 0
        add_tool = request.count if request.kind == "tool" else 0
        committed_input = snap.input_tokens + snap.cache_read_tokens + snap.cache_write_tokens
        rows = (
            ("llm_calls", p.max_llm_calls, snap.llm_calls + snap.out_llm_calls, add_llm),
            ("tool_calls", p.max_tool_calls, snap.tool_calls + snap.out_tool_calls, add_tool),
            (
                "input_tokens",
                p.max_input_tokens,
                committed_input + snap.out_input_tokens,
                reservation.input_tokens,
            ),
            (
                "output_tokens",
                p.max_output_tokens,
                snap.output_tokens + snap.out_output_tokens,
                reservation.output_tokens,
            ),
            (
                "total_tokens",
                p.max_total_tokens,
                snap.total_tokens + snap.out_total_tokens,
                reservation.input_tokens + reservation.output_tokens,
            ),
        )
        for dim, cap, current, add in rows:
            if cap is not None and current + add > cap:
                return BudgetExceeded(dimension=dim, limit=cap, current=current, attempted=add)
        if p.max_cost is not None:
            # Compare in exact Money, mirroring the store's re-validation — a float sum can round
            # just above an on-cap total and spuriously deny an op the store would admit.
            cap = Money.from_usd(p.max_cost)
            committed = snap.cost + snap.out_cost
            if committed + reservation.cost > cap:
                return BudgetExceeded(
                    dimension="cost",
                    limit=p.max_cost,
                    current=committed.to_float(),
                    attempted=reservation.cost.to_float(),
                )
            # Fail closed on an unbounded (unknown) cost under a cost cap: a prior unpriced call
            # already settled (committed_cost excludes it, so the cap could never trip), or this op
            # uses provider-hosted server tools whose charge isn't in the token counts.
            if p.unpriced == "fail_closed":
                if snap.unknown_cost_count > 0:
                    return BudgetExceeded(
                        "a prior call could not be priced — failing closed under a cost cap",
                        dimension="cost",
                        limit=p.max_cost,
                    )
                if request.has_server_tools:
                    return BudgetExceeded(
                        "server tools have unmetered cost — failing closed under a cost cap",
                        dimension="cost",
                        limit=p.max_cost,
                    )
        if p.max_wall_s is not None and snap.elapsed_s > p.max_wall_s:
            return BudgetExceeded(
                dimension="wall_s", limit=p.max_wall_s, current=snap.elapsed_s, attempted=0.0
            )
        return None
