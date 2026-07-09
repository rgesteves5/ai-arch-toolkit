"""Neutral metering primitives — the in-``core`` mechanism that measures usage/cost.

This package knows nothing about "budget": it records facts and runs the
operation lifecycle. Budget *policy* lives in ``toolkit/budget``. See
``docs/internal/metering-plan.md`` for the contract.
"""

from __future__ import annotations

from ai_arch_toolkit.core._metering._admission import (
    AdmissionController,
    AdmissionDecision,
    AdmissionDenied,
    MeterSnapshot,
    NotMeteredOperationError,
    Reservation,
    ResourceLimits,
)
from ai_arch_toolkit.core._metering._cost import Cost, CostKind
from ai_arch_toolkit.core._metering._events import EventStatus, UsageEvent, UsageSink
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._operation import OperationRequest
from ai_arch_toolkit.core._metering._scope import Pricer, RunConfig

__all__ = [
    "AdmissionController",
    "AdmissionDecision",
    "AdmissionDenied",
    "Cost",
    "CostKind",
    "EventStatus",
    "MeterSnapshot",
    "Money",
    "NotMeteredOperationError",
    "OperationRequest",
    "Pricer",
    "Reservation",
    "ResourceLimits",
    "RunConfig",
    "UsageEvent",
    "UsageSink",
]
