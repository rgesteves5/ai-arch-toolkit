"""Exact monetary amounts for metering — opaque, integer pico-USD internally."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Final

__all__ = ["Money"]

_PICO_PER_USD: Final = 1_000_000_000_000  # 1 USD = 1e12 pico-USD


@dataclass(frozen=True, slots=True, order=True, repr=False)
class Money:
    """An exact monetary amount stored as an integer count of pico-USD (1e-12 USD).

    Opaque by design: callers work in USD via :meth:`from_usd` / :meth:`to_float`
    and never see the pico representation. All arithmetic is exact integer
    arithmetic, so accumulating many tiny costs never drifts the way ``float``
    does. Instances are immutable and hashable.
    """

    _pico: int = 0

    @classmethod
    def zero(cls) -> Money:
        """The additive identity, $0."""
        return cls(0)

    @classmethod
    def from_usd(cls, amount: float | Decimal) -> Money:
        """Build from a USD amount, rounding to the nearest pico-USD (banker's rounding).

        Floats are routed through ``Decimal(str(amount))`` so that ``0.10`` means
        exactly ten cents rather than the nearest binary float.
        """
        dec = amount if isinstance(amount, Decimal) else Decimal(str(amount))
        pico = (dec * _PICO_PER_USD).to_integral_value(rounding=ROUND_HALF_EVEN)
        return cls(int(pico))

    @classmethod
    def from_pico(cls, pico: int) -> Money:
        """Build directly from an integer pico-USD count (the pricer's ``rate * tokens`` path)."""
        return cls(int(pico))

    def to_float(self) -> float:
        """USD as a float — for display/serialization only, never further accumulation."""
        return self._pico / _PICO_PER_USD

    @property
    def pico(self) -> int:
        """The raw integer pico-USD count."""
        return self._pico

    def __add__(self, other: Money) -> Money:
        if not isinstance(other, Money):
            return NotImplemented
        return Money(self._pico + other._pico)

    def __radd__(self, other: int) -> Money:
        if other == 0:  # supports sum([...]) which starts from int 0
            return self
        return NotImplemented

    def __sub__(self, other: Money) -> Money:
        if not isinstance(other, Money):
            return NotImplemented
        return Money(self._pico - other._pico)

    def __mul__(self, count: int) -> Money:
        if not isinstance(count, int) or isinstance(count, bool):
            return NotImplemented
        return Money(self._pico * count)

    __rmul__ = __mul__

    def __neg__(self) -> Money:
        return Money(-self._pico)

    def __repr__(self) -> str:
        return f"Money(${self.to_float():.6f})"
