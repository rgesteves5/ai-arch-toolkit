"""Tests for OpenAIModerator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_arch_toolkit.core._exceptions import APIError, RateLimitError


@pytest.fixture
def mock_openai():
    """Patch the openai module used by the moderator."""
    with patch("ai_arch_toolkit.toolkit.moderation._openai.openai") as mock:
        mock.AsyncOpenAI = MagicMock()
        mock.RateLimitError = type("RateLimitError", (Exception,), {})
        mock.APIStatusError = type("APIStatusError", (Exception,), {"status_code": 500})
        yield mock


def _make_moderation_result(flagged: bool, categories: dict, scores: dict):
    """Build a mock moderation API response."""
    result = MagicMock()
    result.flagged = flagged
    result.categories.model_dump.return_value = categories
    result.category_scores.model_dump.return_value = scores

    response = MagicMock()
    response.results = [result]
    return response


class TestOpenAIModerator:
    async def test_unflagged(self, mock_openai) -> None:
        from ai_arch_toolkit.toolkit.moderation._openai import OpenAIModerator

        client = mock_openai.AsyncOpenAI.return_value
        client.moderations.create = AsyncMock(
            return_value=_make_moderation_result(
                flagged=False,
                categories={"violence": False, "hate": False},
                scores={"violence": 0.01, "hate": 0.0},
            )
        )

        mod = OpenAIModerator(api_key="test-key")
        result = await mod.moderate("hello world")

        assert result.flagged is False
        assert result.categories == []
        assert result.scores == {"violence": 0.01}

    async def test_flagged(self, mock_openai) -> None:
        from ai_arch_toolkit.toolkit.moderation._openai import OpenAIModerator

        client = mock_openai.AsyncOpenAI.return_value
        client.moderations.create = AsyncMock(
            return_value=_make_moderation_result(
                flagged=True,
                categories={"violence": True, "hate": False, "harassment": True},
                scores={"violence": 0.95, "hate": 0.02, "harassment": 0.88},
            )
        )

        mod = OpenAIModerator(api_key="test-key")
        result = await mod.moderate("bad content")

        assert result.flagged is True
        assert set(result.categories) == {"violence", "harassment"}
        assert result.scores["violence"] == 0.95
        assert result.raw is not None

    async def test_rate_limit_error(self, mock_openai) -> None:
        from ai_arch_toolkit.toolkit.moderation._openai import OpenAIModerator

        client = mock_openai.AsyncOpenAI.return_value
        client.moderations.create = AsyncMock(
            side_effect=mock_openai.RateLimitError("rate limited")
        )

        mod = OpenAIModerator(api_key="test-key")
        with pytest.raises(RateLimitError):
            await mod.moderate("text")

    async def test_api_error(self, mock_openai) -> None:
        from ai_arch_toolkit.toolkit.moderation._openai import OpenAIModerator

        exc = mock_openai.APIStatusError("server error")
        exc.status_code = 500
        client = mock_openai.AsyncOpenAI.return_value
        client.moderations.create = AsyncMock(side_effect=exc)

        mod = OpenAIModerator(api_key="test-key")
        with pytest.raises(APIError):
            await mod.moderate("text")

    async def test_context_manager(self, mock_openai) -> None:
        from ai_arch_toolkit.toolkit.moderation._openai import OpenAIModerator

        client = mock_openai.AsyncOpenAI.return_value
        client.close = AsyncMock()

        mod = OpenAIModerator(api_key="test-key")
        async with mod as m:
            assert m is mod
        client.close.assert_awaited_once()
