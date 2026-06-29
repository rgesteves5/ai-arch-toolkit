"""Typed cost — a single class with a ``kind`` — replacing ``float | None`` semantics.

``unknown`` is distinct from ``known($0)`` in the type system, so an unpriced
call can never be silently treated as costing zero (the old ``cost or 0.0``
fail-open). The projection keeps a known-sum plus an unknown count separately
and never collapses via :meth:`Cost.merged`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai_arch_toolkit.core._metering._money import Money

__all__ = ["Cost", "CostKind"]

type CostKind = Literal["known", "estimated", "unknown"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Cost:
    """A monetary cost that is ``known``, ``estimated``, or ``unknown``.

    Build via the factories (:meth:`known` / :meth:`estimated` / :meth:`unknown`).
    A ``known``/``estimated`` cost carries an :class:`Money` ``amount`` and no
    ``reason``; an ``unknown`` cost carries a ``reason`` and no ``amount``.
    """

    kind: CostKind
    amount: Money | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "unknown":
            if self.amount is not None:
                raise ValueError("an unknown Cost must not carry an amount")
            if not self.reason:
                raise ValueError("an unknown Cost requires a reason")
        else:
            if self.amount is None:
                raise ValueError(f"a {self.kind} Cost requires an amount")
            if self.reason is not None:
                raise ValueError(f"a {self.kind} Cost must not carry a reason")

    @classmethod
    def known(cls, amount: Money) -> Cost:
        """A priced, actual cost."""
        return cls(kind="known", amount=amount)

    @classmethod
    def estimated(cls, amount: Money) -> Cost:
        """A pre-call worst-case/expected estimate (reservation-only; never settled)."""
        return cls(kind="estimated", amount=amount)

    @classmethod
    def unknown(cls, reason: str) -> Cost:
        """A cost that could not be priced (e.g. an unpriced model)."""
        return cls(kind="unknown", reason=reason)

    @property
    def is_known(self) -> bool:
        """True only for a ``known`` cost."""
        return self.kind == "known"

    @staticmethod
    def merged(*costs: Cost) -> Cost:
        """Combine the costs of a *single composite* operation; certainty degrades.

        If any component is ``unknown`` the result is ``unknown`` (so fail-closed
        cannot be defeated by hiding an unknown inside a sum); otherwise if any is
        ``estimated`` the result is ``estimated``; otherwise ``known``. The
        projection never calls this — it tracks a known-sum + an unknown count.
        """
        if not costs:
            return Cost.known(Money.zero())
        if any(c.kind == "unknown" for c in costs):
            reasons = sorted({c.reason for c in costs if c.kind == "unknown" and c.reason})
            return Cost.unknown("; ".join(reasons) or "merged with an unknown cost")
        total = sum((c.amount for c in costs if c.amount is not None), Money.zero())
        if any(c.kind == "estimated" for c in costs):
            return Cost.estimated(total)
        return Cost.known(total)
