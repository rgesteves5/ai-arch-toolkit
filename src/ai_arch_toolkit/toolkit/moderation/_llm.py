"""LLM-based content moderator — uses any LLM as a classifier."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Literal

from ai_arch_toolkit.core._moderation import ModerationResult
from ai_arch_toolkit.core._sync import _run_sync

if TYPE_CHECKING:
    from ai_arch_toolkit.core._llm import LLM

logger = logging.getLogger(__name__)

_CLASSIFICATION_PROMPT = """\
You are a content moderation classifier. Analyze the following text and determine \
whether it violates any of the listed categories.

Categories: {categories}

Text to analyze:
---
{text}
---

Respond with ONLY a JSON object (no markdown, no extra text):
{{"flagged": true/false, "categories": ["list of violated categories or empty"], \
"explanation": "brief reason"}}"""


class LLMModerator:
    """Moderator that uses any LLM as a content classifier.

    The LLM instance provided should NOT have ``ModerationMiddleware``
    attached, or it will create an infinite moderation loop.

    Example::

        classifier = LLM("claude-haiku-4-5-20251001")
        mod = LLMModerator(classifier, ["Violence", "Harassment", "PII"])
        result = await mod.moderate("some text")
    """

    __slots__ = ("_categories", "_fail_behavior", "_llm")

    def __init__(
        self,
        llm: LLM,
        categories: list[str],
        *,
        fail_behavior: Literal["open", "closed"] = "closed",
    ) -> None:
        self._llm = llm
        self._categories = categories
        self._fail_behavior = fail_behavior

    async def moderate(self, text: str) -> ModerationResult:
        """Classify text using the configured LLM."""
        prompt = _CLASSIFICATION_PROMPT.format(
            categories=", ".join(self._categories),
            text=text,
        )
        try:
            response = await self._llm.complete(prompt, json_mode=True)
            data = json.loads(response.text)
            return ModerationResult(
                flagged=bool(data["flagged"]),
                categories=list(data.get("categories", [])),
                explanation=str(data.get("explanation", "")),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("LLMModerator: failed to parse classifier response: %s", exc)
            return self._fail_result()
        except Exception as exc:
            logger.warning("LLMModerator: classifier LLM error: %s", exc)
            return self._fail_result()

    def moderate_sync(self, text: str) -> ModerationResult:
        """Synchronous wrapper around :meth:`moderate`."""
        return _run_sync(self.moderate(text))

    def _fail_result(self) -> ModerationResult:
        if self._fail_behavior == "closed":
            return ModerationResult(
                flagged=True,
                categories=self._categories,
                explanation="Moderation classifier failed; fail-closed.",
            )
        return ModerationResult(flagged=False, categories=[])
