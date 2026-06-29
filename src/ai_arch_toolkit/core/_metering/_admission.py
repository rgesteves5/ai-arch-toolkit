"""Admission contracts: what a controller decides and what the store re-validates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from ai_arch_toolkit.core._metering._money import Money

if TYPE_CHECKING:
    from ai_arch_toolkit.core._metering._operation import OperationFacts

__all__ = [
    "AdmissionController",
    "AdmissionDecision",
    "AdmissionDenied",
    "MeterSnapshot",
    "NotMeteredOperationError",
    "Reservation",
    "ResourceLimits",
]


class AdmissionDenied(Exception):
    """An operation was denied admission (e.g. a budget cap).

    **Terminal** — never retried or fell back. Neutral to ``core``;
    ``toolkit.budget.BudgetExceeded`` subclasses it. Carries enough structure for
    a report/trace without downcasting to the toolkit type.
    """

    def __init__(
        self,
        message: str = "",
        *,
        dimension: str | None = None,
        limit: float | None = None,
        current: float | None = None,
        attempted: float | None = None,
    ) -> None:
        super().__init__(message or f"admission denied: {dimension or 'limit'}")
        self.dimension = dimension
        self.limit = limit
        self.current = current
        self.attempted = attempted


class NotMeteredOperationError(AdmissionDenied):
    """An unmeterable surface (e.g. batch) was used inside an enforcing scope."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceLimits:
    """The re-validatable caps — the ONLY caps the store guarantees hard under concurrency.

    A custom controller must express its hard caps here; free-form admit logic is
    advisory under races.
    """

    max_llm_calls: int | None = None
    max_tool_calls: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_cost: Money | None = None
    max_wall_s: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Reservation:
    """A controller's optional token/cost holds for one operation (NOT call counts).

    The store always reserves the base llm/tool call count from the operation's
    facts; this carries only the worst-case token/cost holds — empty under the
    soft (``reserve=NONE``) default.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost: Money = field(default_factory=Money.zero)

    @classmethod
    def none(cls) -> Reservation:
        """A zero reservation (the soft default)."""
        return cls()

    def __add__(self, other: Reservation) -> Reservation:
        if not isinstance(other, Reservation):
            return NotImplemented
        return Reservation(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost=self.cost + other.cost,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class MeterSnapshot:
    """An immutable read of the meter — what a controller sees, what reports derive from.

    Committed counts are *started physical attempts*; usage/cost are *settled
    actuals*; ``out_*`` are *reserved but not yet started/settled*. Caps are
    checked against committed + outstanding, never committed alone.
    """

    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost: Money = field(default_factory=Money.zero)
    unknown_cost_count: int = 0
    out_llm_calls: int = 0
    out_tool_calls: int = 0
    out_input_tokens: int = 0
    out_output_tokens: int = 0
    out_cost: Money = field(default_factory=Money.zero)
    elapsed_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Committed input + output + cache_read + cache_write."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    @property
    def out_total_tokens(self) -> int:
        """Outstanding input + output."""
        return self.out_input_tokens + self.out_output_tokens


@dataclass(frozen=True, slots=True, kw_only=True)
class AdmissionDecision:
    """A controller's verdict for one operation: admit (with optional holds) or deny."""

    admitted: bool
    reservation: Reservation = field(default_factory=Reservation)
    limits: ResourceLimits | None = None
    denial: AdmissionDenied | None = None

    def __post_init__(self) -> None:
        if self.admitted and self.denial is not None:
            raise ValueError("an admitted decision must not carry a denial")
        if not self.admitted and self.denial is None:
            raise ValueError("a denied decision requires a denial")

    @classmethod
    def allow(
        cls,
        reservation: Reservation | None = None,
        limits: ResourceLimits | None = None,
    ) -> AdmissionDecision:
        """An admit, optionally with token/cost holds and the caps to re-validate."""
        return cls(admitted=True, reservation=reservation or Reservation(), limits=limits)

    @classmethod
    def deny(cls, denial: AdmissionDenied) -> AdmissionDecision:
        """A denial carrying the (neutral) exception to raise."""
        return cls(admitted=False, denial=denial)


class AdmissionController(Protocol):
    """Decides whether an operation may proceed. PURE: sync, no I/O, no ``await``.

    Implemented in ``toolkit.budget`` and run OUTSIDE the store lock. Hard
    concurrency safety is guaranteed only for caps expressed in the returned
    :class:`ResourceLimits`.
    """

    def admit(self, snapshot: MeterSnapshot, facts: OperationFacts) -> AdmissionDecision: ...
