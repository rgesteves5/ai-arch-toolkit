"""Scope — filters and transforms State layers for Step visibility."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ai_arch_toolkit.core._state import StateSnapshot


@dataclass(frozen=True, slots=True, kw_only=True)
class Scope:
    """Controls what a Step can see in the State."""

    include: frozenset[str] = field(default_factory=frozenset)
    exclude: frozenset[str] = field(default_factory=frozenset)
    transform: dict[str, Callable[[Any], Any]] = field(default_factory=dict)
    enrich: dict[str, Callable[[StateSnapshot], Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.include and self.exclude:
            overlap = self.include & self.exclude
            if overlap:
                raise ValueError(f"include and exclude must not overlap, got: {overlap}")


def apply_scope(snapshot: StateSnapshot, scope: Scope | None) -> StateSnapshot:
    """Apply scope filtering to each layer independently, returning a new snapshot."""
    if scope is None:
        return snapshot

    def _filter_layer(layer: MappingProxyType[str, Any]) -> dict[str, Any]:
        d = dict(layer)
        if scope.include:
            d = {k: v for k, v in d.items() if k in scope.include}
        if scope.exclude:
            d = {k: v for k, v in d.items() if k not in scope.exclude}
        for k in list(d):
            if k in scope.transform:
                d[k] = scope.transform[k](d[k])
        return d

    current = _filter_layer(snapshot.current)
    operational = _filter_layer(snapshot.operational)
    persistent = _filter_layer(snapshot.persistent)
    world = _filter_layer(snapshot.world)

    # Enrich into current layer
    for k, fn in scope.enrich.items():
        current[k] = fn(snapshot)

    return StateSnapshot(
        current=MappingProxyType(current),
        operational=MappingProxyType(operational),
        persistent=MappingProxyType(persistent),
        world=MappingProxyType(world),
    )
