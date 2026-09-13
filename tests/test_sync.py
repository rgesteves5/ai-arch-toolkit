"""Tests for _sync.py — sync wrappers."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import threading
import time
from collections.abc import Callable, Iterator

import pytest

from ai_arch_toolkit.core import _sync as sync_mod
from ai_arch_toolkit.core._sync import _run_sync, _stream_sync, configure_sync_timeouts

_probe: contextvars.ContextVar[str | None] = contextvars.ContextVar("probe", default=None)


def _wait_until(predicate: Callable[[], bool], timeout: float) -> bool:
    """Poll ``predicate`` until it holds or ``timeout`` seconds pass."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.005)
    return True


@pytest.fixture
def restore_sync_timeouts() -> Iterator[None]:
    sync_timeout, join_timeout = sync_mod._sync_timeout, sync_mod._stream_join_timeout
    yield
    configure_sync_timeouts(sync_timeout=sync_timeout, stream_join_timeout=join_timeout)


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


class TestRunSyncTimeout:
    """Inside a running loop the coroutine runs on a worker thread, bounded by ``sync_timeout``."""

    async def test_timeout_cancels_the_coroutine_before_raising(self, restore_sync_timeouts):
        configure_sync_timeouts(sync_timeout=0.2, stream_join_timeout=5.0)
        unwound = threading.Event()
        finished = threading.Event()

        async def slow() -> None:
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                await asyncio.sleep(0.1)  # cleanup on cancellation, e.g. failing a metered op
                unwound.set()
                raise
            finished.set()

        with pytest.raises(TimeoutError, match=r"timed out after 0\.2s"):
            _run_sync(slow())
        # The caller only moves on once the cancelled coroutine has unwound.
        assert unwound.is_set()
        # Well past the moment the 0.5 s sleep would have ended: the body never completes.
        assert not finished.wait(timeout=0.7)

    async def test_timeout_before_the_worker_starts_means_the_coroutine_never_runs(
        self, restore_sync_timeouts, monkeypatch
    ):
        configure_sync_timeouts(sync_timeout=0.05, stream_join_timeout=5.0)
        attach = sync_mod._TaskHandle.attach

        def late_attach(handle: sync_mod._TaskHandle) -> bool:
            time.sleep(0.5)  # the worker's loop only gets going well after the deadline
            return attach(handle)

        monkeypatch.setattr(sync_mod._TaskHandle, "attach", late_attach)
        started = threading.Event()

        async def work() -> None:
            started.set()

        coro = work()
        with pytest.raises(TimeoutError):
            _run_sync(coro)
        assert not started.is_set()
        assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED  # closed, never awaited

    async def test_timeout_bounds_the_wait_for_a_coroutine_that_ignores_cancellation(
        self, restore_sync_timeouts, caplog
    ):
        configure_sync_timeouts(sync_timeout=0.2, stream_join_timeout=0.2)
        cancelled = threading.Event()
        release = threading.Event()

        async def stubborn() -> None:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                while not release.is_set():  # swallows the cancellation and keeps running
                    await asyncio.sleep(0.01)

        started = time.monotonic()
        try:
            with (
                caplog.at_level(logging.WARNING, logger=sync_mod.__name__),
                pytest.raises(TimeoutError),
            ):
                _run_sync(stubborn())
            elapsed = time.monotonic() - started
        finally:
            release.set()
        assert cancelled.wait(timeout=5.0)
        # Bounded by sync_timeout plus the join timeout, not by the coroutine's own duration.
        assert elapsed < 3.0
        assert "still alive" in caplog.text


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

    def test_an_exception_yielded_as_a_value_is_delivered_not_raised(self):
        async def gen():
            yield ValueError("a value")
            yield "after"

        items = list(_stream_sync(lambda: gen()))

        assert isinstance(items[0], ValueError) and str(items[0]) == "a value"
        assert items[1] == "after"

    def test_error_reaches_a_slow_consumer_after_every_item(self, monkeypatch):
        monkeypatch.setattr(sync_mod, "_STREAM_QUEUE_MAXSIZE", 2)

        async def gen():
            for i in range(6):
                yield i
            raise RuntimeError("source failed")

        received: list[int] = []
        with pytest.raises(RuntimeError, match="source failed"):
            for item in _stream_sync(lambda: gen()):
                received.append(item)
                time.sleep(0.02)  # slow consumer: the queue is full when the source fails
        assert received == list(range(6))

    def test_error_after_the_worker_loop_closed_still_reaches_the_consumer(self):
        worker: list[threading.Thread] = []

        async def gen():
            worker.append(threading.current_thread())
            yield 1
            raise ValueError("late failure")

        stream = _stream_sync(lambda: gen())
        assert next(stream) == 1
        # The error is already queued and the worker's loop is gone before the consumer asks.
        assert _wait_until(lambda: not worker[0].is_alive(), timeout=5.0)
        with pytest.raises(ValueError, match="late failure"):
            next(stream)


class TestStreamSyncBackpressure:
    """A slow consumer holds the producer back without ever blocking the worker's event loop."""

    def test_producer_runs_at_most_one_item_past_a_full_queue(self, monkeypatch):
        monkeypatch.setattr(sync_mod, "_STREAM_QUEUE_MAXSIZE", 2)
        produced = 0

        async def gen():
            nonlocal produced
            for i in range(50):
                produced += 1
                yield i

        received: list[int] = []
        lead = 0
        for item in _stream_sync(lambda: gen()):
            received.append(item)
            if len(received) <= 5:
                time.sleep(0.05)  # slow consumer: the producer has every chance to run ahead
                lead = max(lead, produced - len(received))
        assert received == list(range(50))
        # Queue limit plus the one item the producer holds while it waits for room.
        assert lead <= 2 + 1

    def test_full_queue_does_not_block_the_worker_loop(self, monkeypatch):
        monkeypatch.setattr(sync_mod, "_STREAM_QUEUE_MAXSIZE", 1)
        produced = 0
        ticks = 0

        async def heartbeat() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.005)
                ticks += 1

        async def gen():
            nonlocal produced
            beat = asyncio.create_task(heartbeat())
            try:
                for i in range(100):
                    produced += 1
                    yield i
            finally:
                beat.cancel()

        stream = _stream_sync(lambda: gen())
        try:
            assert next(stream) == 0
            # One item consumed, one queued, one held while waiting for room: the queue is full.
            assert _wait_until(lambda: produced >= 3, timeout=5.0)
            before = ticks
            assert _wait_until(lambda: ticks >= before + 5, timeout=5.0)
            assert produced == 3
        finally:
            stream.close()


class TestStreamSyncAbandonment:
    """A consumer that stops early closes the source promptly and reclaims the worker thread."""

    def test_abandoning_cancels_a_pending_source_and_joins_the_worker(self):
        cleaned_up = threading.Event()
        worker: list[threading.Thread] = []

        async def gen():
            worker.append(threading.current_thread())
            try:
                yield 1
                await asyncio.sleep(5)  # the next item is a long way off
                yield 2
            finally:
                cleaned_up.set()

        stream = _stream_sync(lambda: gen())
        assert next(stream) == 1
        started = time.monotonic()
        stream.close()
        assert cleaned_up.wait(timeout=1.0)
        assert _wait_until(lambda: not worker[0].is_alive(), timeout=1.0)
        assert time.monotonic() - started < 1.0

    def test_abandoning_with_a_full_queue_stops_the_producer(self, monkeypatch):
        monkeypatch.setattr(sync_mod, "_STREAM_QUEUE_MAXSIZE", 1)
        cleaned_up = threading.Event()
        worker: list[threading.Thread] = []

        async def gen():
            worker.append(threading.current_thread())
            try:
                i = 0
                while True:  # endless and fast: only the consumer can stop it
                    yield i
                    i += 1
            finally:
                cleaned_up.set()

        stream = _stream_sync(lambda: gen())
        assert next(stream) == 0
        stream.close()
        assert cleaned_up.wait(timeout=1.0)
        assert _wait_until(lambda: not worker[0].is_alive(), timeout=1.0)

    def test_abandoning_does_not_interrupt_a_source_already_cleaning_up(self, monkeypatch):
        monkeypatch.setattr(sync_mod, "_STREAM_QUEUE_MAXSIZE", 1)
        cleanup_started = threading.Event()
        cleaned_up = threading.Event()
        cancel = sync_mod._TaskHandle.cancel

        def cancel_after_cleanup_started(handle: sync_mod._TaskHandle) -> None:
            # The drain saw the consumer leave and is closing the source before the cancel lands.
            assert cleanup_started.wait(timeout=5.0)
            cancel(handle)

        monkeypatch.setattr(sync_mod._TaskHandle, "cancel", cancel_after_cleanup_started)

        async def gen():
            try:
                i = 0
                while True:
                    yield i
                    i += 1
            finally:
                cleanup_started.set()
                await asyncio.sleep(0.2)  # e.g. closing the provider connection
                cleaned_up.set()

        stream = _stream_sync(lambda: gen())
        assert next(stream) == 0
        stream.close()
        assert cleaned_up.wait(timeout=5.0)

    def test_abandoning_frees_a_worker_holding_an_unread_error(self, monkeypatch):
        monkeypatch.setattr(sync_mod, "_STREAM_QUEUE_MAXSIZE", 1)
        failed = threading.Event()
        worker: list[threading.Thread] = []

        async def gen():
            worker.append(threading.current_thread())
            yield 0
            yield 1
            yield 2
            failed.set()
            raise RuntimeError("never read")

        stream = _stream_sync(lambda: gen())
        assert next(stream) == 0
        assert next(stream) == 1
        # Item 2 fills the queue, so the error behind it is still waiting for room.
        assert failed.wait(timeout=5.0)
        stream.close()
        assert _wait_until(lambda: not worker[0].is_alive(), timeout=1.0)

    def test_stream_closed_on_its_own_worker_thread_does_not_join_itself(self):
        # Stands in for the garbage collector finalizing an abandoned stream on the drain thread.
        streams: list[Iterator[int]] = []
        handed_over = threading.Event()
        closed = threading.Event()

        async def gen():
            yield 1
            while not handed_over.is_set():
                await asyncio.sleep(0.005)
            streams[0].close()
            closed.set()
            yield 2

        stream = _stream_sync(lambda: gen())
        streams.append(stream)
        assert next(stream) == 1
        handed_over.set()
        assert closed.wait(timeout=5.0)


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
