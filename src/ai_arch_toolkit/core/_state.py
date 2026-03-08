"""State — layered mutable container with immutable snapshots."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

type MergeStrategy = Literal["last_wins", "collect", "raise"]


class MergeConflictError(KeyError):
    """Raised when parallel steps write the same key with strategy='raise'."""


@dataclass(frozen=True, slots=True, kw_only=True)
class StateSnapshot:
    """Immutable view of all State layers."""

    current: MappingProxyType[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    operational: MappingProxyType[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    persistent: MappingProxyType[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    world: MappingProxyType[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __getitem__(self, key: str) -> Any:
        for layer in (self.current, self.operational, self.persistent, self.world):
            if key in layer:
                return layer[key]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        layers = (self.current, self.operational, self.persistent, self.world)
        return any(key in layer for layer in layers)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def require(self, key: str) -> Any:
        """Get a key or raise a helpful KeyError listing available keys."""
        try:
            return self[key]
        except KeyError:
            available = sorted(self.keys())
            raise KeyError(
                f"Key {key!r} not found in state. Available keys: {available}"
            ) from None

    def keys(self) -> frozenset[str]:
        """Union of all layer keys."""
        return frozenset(
            set(self.current) | set(self.operational) | set(self.persistent) | set(self.world)
        )

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            "current": dict(self.current),
            "operational": dict(self.operational),
            "persistent": dict(self.persistent),
            "world": dict(self.world),
        }

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, Any]]) -> StateSnapshot:
        return cls(
            current=MappingProxyType(data.get("current", {})),
            operational=MappingProxyType(data.get("operational", {})),
            persistent=MappingProxyType(data.get("persistent", {})),
            world=MappingProxyType(data.get("world", {})),
        )


class State:
    """Mutable container with four named layers; produces immutable snapshots."""

    __slots__ = ("_current", "_operational", "_persistent", "_world")

    def __init__(
        self,
        current: dict[str, Any] | None = None,
        operational: dict[str, Any] | None = None,
        persistent: dict[str, Any] | None = None,
        world: dict[str, Any] | None = None,
    ) -> None:
        self._current: dict[str, Any] = current or {}
        self._operational: dict[str, Any] = operational or {}
        self._persistent: dict[str, Any] = persistent or {}
        self._world: dict[str, Any] = world or {}

    # --- Read access (layered lookup) ---

    def __getitem__(self, key: str) -> Any:
        for layer in (self._current, self._operational, self._persistent, self._world):
            if key in layer:
                return layer[key]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return any(
            key in layer
            for layer in (self._current, self._operational, self._persistent, self._world)
        )

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def require(self, key: str) -> Any:
        """Get a key or raise a helpful KeyError listing available keys."""
        try:
            return self[key]
        except KeyError:
            available = sorted(self.keys())
            raise KeyError(
                f"Key {key!r} not found in state. Available keys: {available}"
            ) from None

    # --- Write access ---

    def __setitem__(self, key: str, value: Any) -> None:
        self._operational[key] = value

    def set(self, key: str, value: Any, *, layer: str = "operational") -> None:
        """Write to a specific layer."""
        target = self._get_layer(layer)
        target[key] = value

    # --- Layer accessors (return copies) ---

    @property
    def current(self) -> dict[str, Any]:
        return dict(self._current)

    @property
    def operational(self) -> dict[str, Any]:
        return dict(self._operational)

    @property
    def persistent(self) -> dict[str, Any]:
        return dict(self._persistent)

    @property
    def world(self) -> dict[str, Any]:
        return dict(self._world)

    def keys(self) -> frozenset[str]:
        """Union of all layer keys."""
        return frozenset(
            set(self._current) | set(self._operational) | set(self._persistent) | set(self._world)
        )

    # --- Snapshot, fork, merge ---

    def snapshot(self) -> StateSnapshot:
        """Immutable copy of all layers."""
        return StateSnapshot(
            current=MappingProxyType(dict(self._current)),
            operational=MappingProxyType(dict(self._operational)),
            persistent=MappingProxyType(dict(self._persistent)),
            world=MappingProxyType(dict(self._world)),
        )

    def fork(self) -> State:
        """Deep copy current/operational/persistent; world shared by reference."""
        return State(
            current=copy.deepcopy(self._current),
            operational=copy.deepcopy(self._operational),
            persistent=copy.deepcopy(self._persistent),
            world=self._world,
        )

    def merge(
        self,
        *results: Any,
        strategy: MergeStrategy = "last_wins",
    ) -> None:
        """Fold Result artifacts into operational layer.

        Args:
            results: Result objects whose `artifacts` dicts are merged.
            strategy: Conflict resolution — 'last_wins', 'collect', or 'raise'.
        """
        from ai_arch_toolkit.core._step import Result

        all_keys: dict[str, list[Any]] = {}
        for r in results:
            if not isinstance(r, Result):
                raise TypeError(f"Expected Result, got {type(r).__name__}")
            for k, v in r.artifacts.items():
                all_keys.setdefault(k, []).append(v)

        for key, values in all_keys.items():
            if len(values) == 1 or strategy == "last_wins":
                self._operational[key] = values[-1]
            elif strategy == "collect":
                existing = self._operational.get(key)
                if isinstance(existing, list):
                    existing.extend(values)
                else:
                    self._operational[key] = values
            elif strategy == "raise":
                if len(values) > 1:
                    raise MergeConflictError(f"Conflict on key {key!r}: {len(values)} values")
                self._operational[key] = values[-1]

    # --- Serialization ---

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            "current": dict(self._current),
            "operational": dict(self._operational),
            "persistent": dict(self._persistent),
            "world": dict(self._world),
        }

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, Any]]) -> State:
        return cls(
            current=data.get("current", {}),
            operational=data.get("operational", {}),
            persistent=data.get("persistent", {}),
            world=data.get("world", {}),
        )

    @classmethod
    def from_trace(cls, trace: Any) -> State:
        """Reconstruct initial state from a Trace. Does not replay execution."""
        from ai_arch_toolkit.core._trace import Trace

        if not isinstance(trace, Trace):
            raise TypeError(f"Expected Trace, got {type(trace).__name__}")
        return cls.from_dict(trace.initial_state)

    # --- Internal ---

    def _get_layer(self, name: str) -> dict[str, Any]:
        layers = {
            "current": self._current,
            "operational": self._operational,
            "persistent": self._persistent,
            "world": self._world,
        }
        if name not in layers:
            raise ValueError(f"Unknown layer {name!r}. Must be one of {list(layers)}")
        return layers[name]
