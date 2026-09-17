"""ReasoningSpec — declarative description of how a single agent reasons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._response import OutputSchema
from ai_arch_toolkit.core._retry import RetryConfig
from ai_arch_toolkit.core._trace import TRACE_CAPTURE_MODES, TraceCapture

__all__ = ["ReasoningSpec"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ReasoningSpec:
    """Declarative description of how one agent reasons.

    Carries only the reasoning structure — the named ``strategy``, system
    prompt, and limits — not the model or tools, which are runtime objects
    supplied to ``build_flow``. ``knobs`` holds strategy-specific options.
    ``trace_capture`` sets what the compiled flow's trace records (see ``Flow``).
    """

    strategy: str = "react"
    system: str = ""
    max_iterations: int = 10
    knobs: Mapping[str, Any] = field(default_factory=dict)
    policy: Policy | None = None
    timeout: float | None = None
    trace_capture: TraceCapture = "keys"
    llm_kwargs: Mapping[str, Any] = field(default_factory=dict)
    output_schema: OutputSchema | type | None = None

    def __post_init__(self) -> None:
        if self.trace_capture not in TRACE_CAPTURE_MODES:
            msg = f"trace_capture must be 'keys', 'full' or 'none', got {self.trace_capture!r}"
            raise ValueError(msg)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ReasoningSpec:
        """Build a spec from a plain mapping (e.g. parsed JSON/YAML/dict).

        ``policy`` may be a ``Policy`` or a mapping of its fields (``retry`` a mapping of
        ``RetryConfig`` fields); ``output_schema`` a class, an ``OutputSchema``, or a mapping with
        a ``schema``. Nothing is dropped silently: an unknown key, or a value that cannot be used,
        raises ``ValueError`` naming it.
        """
        unknown = sorted(set(data) - _SPEC_KEYS)
        if unknown:
            raise ValueError(
                f"unknown ReasoningSpec key(s) {unknown}; expected any of {sorted(_SPEC_KEYS)}"
            )
        return cls(
            strategy=str(data.get("strategy", "react")),
            system=str(data.get("system", "")),
            max_iterations=int(data.get("max_iterations", 10)),
            knobs=dict(data.get("knobs") or {}),
            policy=_coerce_policy(data.get("policy")),
            timeout=data.get("timeout"),
            trace_capture=data.get("trace_capture", "keys"),
            llm_kwargs=dict(data.get("llm_kwargs") or {}),
            output_schema=_coerce_output_schema(data.get("output_schema")),
        )


_SPEC_KEYS = frozenset(
    {
        "strategy",
        "system",
        "max_iterations",
        "knobs",
        "policy",
        "timeout",
        "trace_capture",
        "llm_kwargs",
        "output_schema",
    }
)
# ``fallback`` is a Step: it has no mapping form, so it is set on a ``Policy`` in code.
_POLICY_KEYS = frozenset(f.name for f in fields(Policy)) - {"fallback"}


def _coerce_policy(value: Any) -> Policy | None:
    if value is None or isinstance(value, Policy):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"policy must be a Policy or a mapping, got {type(value).__name__}")
    unknown = sorted(set(value) - _POLICY_KEYS)
    if unknown:
        raise ValueError(
            f"unknown policy key(s) {unknown}; a mapping accepts {sorted(_POLICY_KEYS)}"
        )
    options = dict(value)
    retry = options.get("retry")
    if isinstance(retry, Mapping):
        try:
            options["retry"] = RetryConfig(**retry)
        except TypeError as exc:
            raise ValueError(f"policy retry: {exc}") from exc
    elif retry is not None and not isinstance(retry, RetryConfig):
        raise ValueError(f"policy retry must be a mapping, got {type(retry).__name__}")
    return Policy(**options)


def _coerce_output_schema(value: Any) -> OutputSchema | type | None:
    if value is None or isinstance(value, (OutputSchema, type)):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(
            "output_schema must be a class, an OutputSchema or a mapping, "
            f"got {type(value).__name__}"
        )
    schema = value.get("schema")
    if not isinstance(schema, Mapping):
        raise ValueError("output_schema mapping needs a 'schema' mapping")
    return OutputSchema(
        name=str(value.get("name", "output")),
        schema=dict(schema),
        strict=bool(value.get("strict", True)),
    )
