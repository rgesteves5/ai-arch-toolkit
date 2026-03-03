"""Pipeline types — PipelineContext, PhaseResult, PipelineResult."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import field
from types import MappingProxyType
from typing import Any, Literal

type PhaseFn = Callable[[PipelineContext], Awaitable[PhaseResult]]
type PhaseStatus = Literal["ok", "partial", "failed", "skipped"]

_STATUS_RANK: dict[str, int] = {"ok": 0, "partial": 1, "failed": 2}


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PhaseResult:
    """Immutable output of a single pipeline phase."""

    status: PhaseStatus
    phase: str = ""
    duration: float = 0.0
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    token_usage: dict[str, int] | None = None
    warnings: list[str] = field(default_factory=list)

    # --- Factory methods ---

    @classmethod
    def ok(
        cls,
        *,
        phase: str = "",
        token_usage: dict[str, int] | None = None,
        warnings: list[str] | None = None,
        **artifacts: Any,
    ) -> PhaseResult:
        return cls(
            status="ok",
            phase=phase,
            artifacts=artifacts,
            token_usage=token_usage,
            warnings=warnings or [],
        )

    @classmethod
    def partial(
        cls,
        *,
        phase: str = "",
        error: str | None = None,
        token_usage: dict[str, int] | None = None,
        warnings: list[str] | None = None,
        **artifacts: Any,
    ) -> PhaseResult:
        return cls(
            status="partial",
            phase=phase,
            error=error,
            artifacts=artifacts,
            token_usage=token_usage,
            warnings=warnings or [],
        )

    @classmethod
    def failed(
        cls,
        error: str,
        *,
        phase: str = "",
        token_usage: dict[str, int] | None = None,
        warnings: list[str] | None = None,
    ) -> PhaseResult:
        return cls(
            status="failed",
            phase=phase,
            error=error,
            token_usage=token_usage,
            warnings=warnings or [],
        )

    @classmethod
    def skipped(cls, *, phase: str = "", reason: str = "") -> PhaseResult:
        return cls(
            status="skipped",
            phase=phase,
            error=reason or None,
        )

    # --- Properties ---

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"

    @property
    def is_partial(self) -> bool:
        return self.status == "partial"

    @property
    def is_failure(self) -> bool:
        return self.status == "failed"

    @property
    def is_skipped(self) -> bool:
        return self.status == "skipped"

    @property
    def has_artifacts(self) -> bool:
        return bool(self.artifacts)

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "status": self.status,
            "phase": self.phase,
            "duration": self.duration,
        }
        if self.artifacts:
            d["artifacts"] = self.artifacts
        if self.error is not None:
            d["error"] = self.error
        if self.token_usage is not None:
            d["token_usage"] = self.token_usage
        if self.warnings:
            d["warnings"] = self.warnings
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhaseResult:
        return cls(
            status=data["status"],
            phase=data.get("phase", ""),
            duration=data.get("duration", 0.0),
            artifacts=data.get("artifacts", {}),
            error=data.get("error"),
            token_usage=data.get("token_usage"),
            warnings=data.get("warnings", []),
        )


class PipelineContext:
    """Mutable accumulator for pipeline state. Dict-like access to artifacts."""

    __slots__ = ("_data", "_metadata", "_phase_results", "_provenance")

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._data: dict[str, Any] = dict(data) if data else {}
        self._metadata: dict[str, Any] = dict(metadata) if metadata else {}
        self._provenance: dict[str, str] = {}
        self._phase_results: dict[str, PhaseResult] = {}

    # --- Dict-like access ---

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def require(self, key: str) -> Any:
        """Get artifact or raise KeyError with helpful message."""
        if key not in self._data:
            available = ", ".join(sorted(self._data.keys())) or "(none)"
            msg = f"Required artifact {key!r} not found. Available: {available}"
            raise KeyError(msg)
        return self._data[key]

    # --- Properties ---

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def provenance(self) -> dict[str, str]:
        return dict(self._provenance)

    # --- Merge ---

    def merge(self, artifacts: dict[str, Any], *, phase: str = "") -> None:
        """Bulk set artifacts with provenance. Warns on cross-phase overwrite."""
        for key, value in artifacts.items():
            if key in self._provenance and self._provenance[key] != phase:
                warnings.warn(
                    f"Artifact {key!r} overwritten: {self._provenance[key]!r} -> {phase!r}",
                    stacklevel=2,
                )
            self._data[key] = value
            if phase:
                self._provenance[key] = phase

    # --- Phase accumulation ---

    @property
    def phase_results(self) -> Mapping[str, PhaseResult]:
        return MappingProxyType(self._phase_results)

    def _record_phase(self, name: str, result: PhaseResult) -> None:
        self._phase_results[name] = result

    @property
    def total_token_usage(self) -> dict[str, int] | None:
        totals: dict[str, int] = {}
        for result in self._phase_results.values():
            if result.token_usage:
                for k, v in result.token_usage.items():
                    totals[k] = totals.get(k, 0) + v
        return totals or None

    @property
    def total_warnings(self) -> list[str]:
        out: list[str] = []
        for result in self._phase_results.values():
            out.extend(result.warnings)
        return out

    @property
    def total_duration(self) -> float:
        return sum(r.duration for r in self._phase_results.values())

    @property
    def last_phase(self) -> str | None:
        if not self._phase_results:
            return None
        return next(reversed(self._phase_results))

    @property
    def last_result(self) -> PhaseResult | None:
        if not self._phase_results:
            return None
        return next(reversed(self._phase_results.values()))

    def produced_by(self, key: str) -> str | None:
        return self._provenance.get(key)

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": self._data,
            "metadata": self._metadata,
            "provenance": self._provenance,
            "phase_results": {k: v.to_dict() for k, v in self._phase_results.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineContext:
        ctx = cls(data=data.get("data"), metadata=data.get("metadata"))
        ctx._provenance = dict(data.get("provenance", {}))
        for name, rd in data.get("phase_results", {}).items():
            ctx._phase_results[name] = PhaseResult.from_dict(rd)
        return ctx


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PipelineResult:
    """Aggregated result of a full pipeline run."""

    status: PhaseStatus
    phases: tuple[PhaseResult, ...] = ()
    context: PipelineContext | None = None
    duration: float = 0.0
    total_token_usage: dict[str, int] | None = None
    total_warnings: list[str] = field(default_factory=list)

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"

    @property
    def failed_phases(self) -> list[str]:
        return [p.phase for p in self.phases if p.is_failure]

    @property
    def completed_phases(self) -> list[str]:
        return [p.phase for p in self.phases if p.is_ok]

    @property
    def skipped_phases(self) -> list[str]:
        return [p.phase for p in self.phases if p.is_skipped]
