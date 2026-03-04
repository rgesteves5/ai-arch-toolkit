"""Tests for _sync.py — sync wrappers."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._sync import _run_sync, _stream_sync


class TestRunSync:
    def test_runs_coroutine(self):
        async def add(a: int, b: int) -> int:
            return a + b

        assert _run_sync(add(2, 3)) == 5

    def test_returns_none(self):
        async def noop() -> None:
            pass

        assert _run_sync(noop()) is None

    def test_propagates_exception(self):
        async def fail() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            _run_sync(fail())


class TestStreamSync:
    def test_yields_items(self):
        async def gen():
            for i in range(5):
                yield i

        result = list(_stream_sync(lambda: gen()))
        assert result == [0, 1, 2, 3, 4]

    def test_empty_iterator(self):
        async def gen():
            return
            yield

        result = list(_stream_sync(lambda: gen()))
        assert result == []

    def test_propagates_exception(self):
        async def gen():
            yield 1
            raise RuntimeError("stream error")

        with pytest.raises(RuntimeError, match="stream error"):
            list(_stream_sync(lambda: gen()))
