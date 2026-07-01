"""Tests for _sync.py — sync wrappers."""

from __future__ import annotations

import contextvars

import pytest

from ai_arch_toolkit.core._sync import _run_sync, _stream_sync

_probe: contextvars.ContextVar[str | None] = contextvars.ContextVar("probe", default=None)


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


class TestContextPropagation:
    """The metering scope rides a ContextVar; sync wrappers that hop threads must carry it."""

    def test_run_sync_direct_path_sees_the_contextvar(self):
        # No running loop -> asyncio.run in this thread; context is naturally present.
        _probe.set("direct")

        async def read() -> str | None:
            return _probe.get()

        assert _run_sync(read()) == "direct"

    async def test_run_sync_thread_path_carries_the_contextvar(self):
        # Inside a running loop -> _run_sync spawns a fresh thread; without copy_context the
        # coroutine would see the default (None) instead of the bound value.
        token = _probe.set("threaded")
        try:

            async def read() -> str | None:
                return _probe.get()

            assert _run_sync(read()) == "threaded"
        finally:
            _probe.reset(token)

    def test_stream_sync_carries_the_contextvar(self):
        _probe.set("streamed")

        async def gen():
            yield _probe.get()

        assert list(_stream_sync(lambda: gen())) == ["streamed"]
