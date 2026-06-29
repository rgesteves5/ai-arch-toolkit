"""The request a charge site submits to the meter (the operation handle comes with the store)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ai_arch_toolkit.core._metering._cost import Cost
    from ai_arch_toolkit.core._metering._store import MeterStore
    from ai_arch_toolkit.core._response import Usage

__all__ = ["MeterOperation", "OperationRequest"]


@dataclass(frozen=True, slots=True, kw_only=True)
class OperationRequest:
    """Pure FACTS about an operation, built by ``core`` *after* middleware ``before``.

    Carries no estimates and no heuristics — the controller's injected estimator
    turns these into a token/cost estimate at admit time, so estimation opinion
    stays in ``toolkit``. ``metadata`` is low-cardinality scalars only (never raw
    prompts/secrets/PII); the store redacts before persist/emit.
    """

    kind: Literal["llm", "tool", "custom"]
    parent_span_id: str
    count: int = 1
    mode: Literal["complete", "stream"] | None = None
    model: str | None = None
    provider: str | None = None
    declared_max_output_tokens: int | None = None
    content_size_hint: int | None = None
    non_text_parts: int = 0
    has_server_tools: bool = False
    metadata: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # llm/tool ops always consume exactly their call count; custom ops never touch
        # the call caps (count must be 0) — so an LLM call can't be laundered as count=0.
        if self.kind in ("llm", "tool"):
            if self.count < 1:
                raise ValueError(f"a {self.kind} operation requires count >= 1, got {self.count}")
        elif self.count != 0:
            raise ValueError(f"a custom operation must have count == 0, got {self.count}")
        if self.mode is not None and self.kind != "llm":
            raise ValueError("mode is only valid for kind='llm'")


class MeterOperation:
    """A stateless handle to one in-flight operation; delegates to the store by ``op_id``.

    Lifecycle: :meth:`mark_started` commits the call count, then :meth:`settle`
    records actuals on success or :meth:`fail` keeps the count on error.
    :meth:`abort` fully releases an operation that never started.
    """

    __slots__ = ("_op_id", "_store")

    def __init__(self, store: MeterStore, op_id: str) -> None:
        self._store = store
        self._op_id = op_id

    @property
    def op_id(self) -> str:
        """This operation's unique id within the run."""
        return self._op_id

    def mark_started(self) -> None:
        """Commit the call count — the operation reached the provider/tool."""
        self._store.mark_started(self._op_id)

    def settle(self, *, usage: Usage, cost: Cost) -> None:
        """Record the actual usage and cost (idempotent; ``cost`` must not be estimated)."""
        self._store.settle(self._op_id, usage=usage, cost=cost)

    def fail(self) -> None:
        """A started operation errored — the count stays, holds are released."""
        self._store.fail(self._op_id)

    def abort(self) -> None:
        """A never-started operation is fully released (no count)."""
        self._store.abort(self._op_id)
