"""OpenAI Moderation API moderator."""

from __future__ import annotations

from typing import Any

from ai_arch_toolkit.core._exceptions import APIError, RateLimitError
from ai_arch_toolkit.core._moderation import ModerationResult
from ai_arch_toolkit.core._providers import OWN_BASE_URLS
from ai_arch_toolkit.core._providers._base import LoopAwareClientCache
from ai_arch_toolkit.core._providers._imports import require_sdk
from ai_arch_toolkit.core._sync import _run_sync

with require_sdk("openai"):
    import openai


class OpenAIModerator(LoopAwareClientCache):
    """Moderator backed by OpenAI's free moderation endpoint.

    Uses ``omni-moderation-latest`` by default which supports text
    classification across violence, harassment, self-harm, sexual,
    hate, and other categories.

    Example::

        mod = OpenAIModerator()
        result = await mod.moderate("some text")
        if result.flagged:
            print(result.categories)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "omni-moderation-latest",
    ) -> None:
        # The endpoint is always given: left out, the SDK reads OPENAI_BASE_URL and would send
        # the key there (D48). The client is built on first use, in the call's event loop, and
        # rebuilt when a sync call's loop has closed (G-17).
        base_url = OWN_BASE_URLS["openai"]
        self._install_client(lambda: openai.AsyncOpenAI(api_key=api_key, base_url=base_url))
        self._model = model

    async def moderate(self, text: str) -> ModerationResult:
        """Check text against OpenAI's moderation endpoint."""
        try:
            response = await self._client.moderations.create(model=self._model, input=text)
        except openai.RateLimitError as exc:
            raise RateLimitError(429, str(exc)) from exc
        except openai.APIStatusError as exc:
            raise APIError(exc.status_code, str(exc)) from exc

        result = response.results[0]
        cats_dict: dict[str, bool] = result.categories.model_dump()
        scores_dict: dict[str, float] = result.category_scores.model_dump()

        flagged_cats = [name for name, val in cats_dict.items() if val]
        nonzero_scores = {name: score for name, score in scores_dict.items() if score > 0}

        return ModerationResult(
            flagged=result.flagged,
            categories=flagged_cats,
            scores=nonzero_scores,
            raw=response,
        )

    def moderate_sync(self, text: str) -> ModerationResult:
        """Synchronous wrapper around :meth:`moderate`."""
        return _run_sync(self.moderate(text))

    async def __aenter__(self) -> OpenAIModerator:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
