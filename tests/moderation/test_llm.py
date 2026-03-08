"""Tests for LLMModerator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_arch_toolkit.toolkit.moderation._llm import LLMModerator


def _make_response(text: str) -> MagicMock:
    r = MagicMock()
    r.text = text
    return r


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.complete = AsyncMock()
    return llm


CATEGORIES = ["Violence", "Harassment", "PII"]


class TestLLMModerator:
    async def test_unflagged(self, mock_llm) -> None:
        mock_llm.complete.return_value = _make_response(
            json.dumps({"flagged": False, "categories": [], "explanation": "all clear"})
        )
        mod = LLMModerator(mock_llm, CATEGORIES)
        result = await mod.moderate("hello world")

        assert result.flagged is False
        assert result.categories == []
        assert result.explanation == "all clear"

    async def test_flagged(self, mock_llm) -> None:
        mock_llm.complete.return_value = _make_response(
            json.dumps(
                {
                    "flagged": True,
                    "categories": ["Violence"],
                    "explanation": "contains violence",
                }
            )
        )
        mod = LLMModerator(mock_llm, CATEGORIES)
        result = await mod.moderate("violent text")

        assert result.flagged is True
        assert result.categories == ["Violence"]

    async def test_fail_closed_on_parse_error(self, mock_llm) -> None:
        mock_llm.complete.return_value = _make_response("not json at all")
        mod = LLMModerator(mock_llm, CATEGORIES, fail_behavior="closed")
        result = await mod.moderate("text")

        assert result.flagged is True
        assert result.categories == CATEGORIES

    async def test_fail_open_on_parse_error(self, mock_llm) -> None:
        mock_llm.complete.return_value = _make_response("not json")
        mod = LLMModerator(mock_llm, CATEGORIES, fail_behavior="open")
        result = await mod.moderate("text")

        assert result.flagged is False
        assert result.categories == []

    async def test_fail_closed_on_llm_exception(self, mock_llm) -> None:
        mock_llm.complete.side_effect = RuntimeError("LLM down")
        mod = LLMModerator(mock_llm, CATEGORIES, fail_behavior="closed")
        result = await mod.moderate("text")

        assert result.flagged is True

    async def test_fail_open_on_llm_exception(self, mock_llm) -> None:
        mock_llm.complete.side_effect = RuntimeError("LLM down")
        mod = LLMModerator(mock_llm, CATEGORIES, fail_behavior="open")
        result = await mod.moderate("text")

        assert result.flagged is False

    async def test_categories_in_prompt(self, mock_llm) -> None:
        mock_llm.complete.return_value = _make_response(
            json.dumps({"flagged": False, "categories": [], "explanation": ""})
        )
        mod = LLMModerator(mock_llm, ["Cat1", "Cat2"])
        await mod.moderate("text")

        prompt = mock_llm.complete.call_args[0][0]
        assert "Cat1" in prompt
        assert "Cat2" in prompt

    async def test_missing_key_fails_closed(self, mock_llm) -> None:
        mock_llm.complete.return_value = _make_response(json.dumps({"wrong_key": True}))
        mod = LLMModerator(mock_llm, CATEGORIES, fail_behavior="closed")
        result = await mod.moderate("text")

        assert result.flagged is True
