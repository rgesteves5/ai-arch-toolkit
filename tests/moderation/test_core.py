"""Tests for core moderation types and protocol."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._moderation import ModerationError, ModerationResult, Moderator


class TestModerationResult:
    def test_fields(self) -> None:
        r = ModerationResult(
            flagged=True,
            categories=["violence"],
            scores={"violence": 0.95},
            explanation="violent content",
            raw={"debug": True},
        )
        assert r.flagged is True
        assert r.categories == ["violence"]
        assert r.scores == {"violence": 0.95}
        assert r.explanation == "violent content"
        assert r.raw == {"debug": True}

    def test_defaults(self) -> None:
        r = ModerationResult(flagged=False, categories=[])
        assert r.scores == {}
        assert r.explanation == ""
        assert r.raw is None

    def test_frozen(self) -> None:
        r = ModerationResult(flagged=False, categories=[])
        with pytest.raises(AttributeError):
            r.flagged = True  # type: ignore[misc]


class TestModerationError:
    def test_carries_fields(self) -> None:
        err = ModerationError(["violence", "hate"], "bad content")
        assert err.categories == ["violence", "hate"]
        assert err.explanation == "bad content"

    def test_str_with_explanation(self) -> None:
        err = ModerationError(["violence"], "too violent")
        assert "violence" in str(err)
        assert "too violent" in str(err)

    def test_str_without_explanation(self) -> None:
        err = ModerationError(["hate"])
        assert "hate" in str(err)

    def test_is_exception(self) -> None:
        with pytest.raises(ModerationError):
            raise ModerationError(["test"])


class TestModeratorProtocol:
    def test_isinstance_check(self) -> None:
        class MyModerator:
            async def moderate(self, text: str) -> ModerationResult:
                return ModerationResult(flagged=False, categories=[])

        assert isinstance(MyModerator(), Moderator)

    def test_non_conforming(self) -> None:
        class NotAModerator:
            pass

        assert not isinstance(NotAModerator(), Moderator)
