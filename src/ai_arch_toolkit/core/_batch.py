"""Batch API types for bulk LLM requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_arch_toolkit.core._response import Response


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchRequest:
    """A single request in a batch submission."""

    messages: list[dict[str, Any]]
    system: str | None = None
    tools: list[dict[str, Any]] | None = None
    custom_id: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchResult:
    """Result of a single request in a batch."""

    custom_id: str
    response: Response | None = None
    error: str | None = None
