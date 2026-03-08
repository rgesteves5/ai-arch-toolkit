"""Tests for Step and Result."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._response import Usage
from ai_arch_toolkit.core._state import StateSnapshot
from ai_arch_toolkit.core._step import Result, Step


class TestResult:
    def test_defaults(self) -> None:
        r = Result()
        assert r.value is None
        assert r.artifacts == {}
        assert r.is_ok
        assert not r.is_error
        assert r.cost == 0.0
        assert r.confidence is None

    def test_error_result(self) -> None:
        r = Result(error="something broke")
        assert r.is_error
        assert not r.is_ok

    def test_with_artifacts(self) -> None:
        r = Result(value=42, artifacts={"key": "val"}, cost=0.5, confidence=0.9)
        assert r.value == 42
        assert r.artifacts["key"] == "val"
        assert r.cost == 0.5
        assert r.confidence == 0.9

    def test_roundtrip(self) -> None:
        r = Result(
            value="hello",
            artifacts={"a": 1},
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=0.1,
            confidence=0.8,
            duration=1.5,
        )
        d = r.to_dict()
        r2 = Result.from_dict(d)
        assert r2.value == "hello"
        assert r2.artifacts == {"a": 1}
        assert r2.usage.input_tokens == 10
        assert r2.cost == 0.1
        assert r2.confidence == 0.8
        assert r2.duration == 1.5

    def test_from_dict_defaults(self) -> None:
        r = Result.from_dict({})
        assert r.value is None
        assert r.is_ok


class TestStep:
    def test_creation(self) -> None:
        async def my_fn(snap: StateSnapshot) -> Result:
            return Result(value="done")

        step = Step(name="test", fn=my_fn)
        assert step.name == "test"
        assert step.policy is None
        assert step.fallback is None

    def test_empty_name_raises(self) -> None:
        async def noop(snap: StateSnapshot) -> Result:
            return Result()

        with pytest.raises(ValueError, match="non-empty"):
            Step(name="", fn=noop)

    def test_with_fallback(self) -> None:
        async def primary(snap: StateSnapshot) -> Result:
            return Result(error="fail")

        async def backup(snap: StateSnapshot) -> Result:
            return Result(value="recovered")

        fb = Step(name="backup", fn=backup)
        step = Step(name="primary", fn=primary, fallback=fb)
        assert step.fallback is not None
        assert step.fallback.name == "backup"
