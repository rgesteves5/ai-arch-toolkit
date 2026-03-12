"""Moderator protocol and core types for content moderation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True, kw_only=True)
class ModerationResult:
    """Result of a moderation check.

    Attributes:
        flagged: Whether the content was flagged as violating any category.
        categories: List of category names that were flagged.
        scores: Per-category confidence scores (optional, provider-dependent).
        explanation: Human-readable explanation of the moderation decision.
        raw: Provider-specific response object for debugging.
    """

    flagged: bool
    categories: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    raw: Any = None


class ModerationError(Exception):
    """Raised when content is flagged by a moderator."""

    def __init__(self, categories: list[str], explanation: str = "") -> None:
        self.categories = categories
        self.explanation = explanation
        detail = ", ".join(categories)
        msg = f"Content flagged: [{detail}]"
        if explanation:
            msg += f" — {explanation}"
        super().__init__(msg)


@runtime_checkable
class Moderator(Protocol):
    """Protocol for content moderators.

    Implementations must provide an async ``moderate`` method.
    Concrete classes may add ``moderate_sync`` for synchronous use.
    """

    async def moderate(self, text: str) -> ModerationResult:
        """Check text for policy violations."""
        ...
