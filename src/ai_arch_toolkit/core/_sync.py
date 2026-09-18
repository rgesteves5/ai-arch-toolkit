"""Safe sync wrappers for async code."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import os
import threading
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from queue import Full, Queue
from typing import Any, cast

from ai_arch_toolkit.core._stream_lifecycle import close_async

logger = logging.getLogger(__name__)

# Configurable timeout defaults (seconds) — overridable via env vars or configure_sync_timeouts()


def _read_positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number of seconds, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive number of seconds, got {raw!r}")
    return value


_sync_timeout: float = _read_positive_float_env("AI_ARCH_SYNC_TIMEOUT", 300.0)
_stream_join_timeout: float = _read_positive_float_env("AI_ARCH_STREAM_JOIN_TIMEOUT", 5.0)

_SENTINEL = object()


class _SourceError:
    """An exception the stream's source raised, as opposed to one it yielded as a value."""

    __slots__ = ("error",)

    def __init__(self, error: BaseException) -> None:
        self.error = error


# A sync stream buffers at most this many items ahead of its consumer. A producer facing a full
# queue sleeps this many seconds between offers, so the worker's event loop keeps running.
_STREAM_QUEUE_MAXSIZE = 256
_STREAM_PUT_INTERVAL = 0.005


def configure_sync_timeouts(
    sync_timeout: float | None = None,
    stream_join_timeout: float | None = None,
) -> None:
    """Configure sync wrapper timeouts.

    Defaults come from ``AI_ARCH_SYNC_TIMEOUT`` (300) and ``AI_ARCH_STREAM_JOIN_TIMEOUT`` (5).

    Args:
        sync_timeout: Seconds a sync call made while an event loop is already running may take
            before its coroutine is cancelled and ``TimeoutError`` is raised.
        stream_join_timeout: Seconds to wait for a sync wrapper's worker thread to finish once a
            sync stream ends or is abandoned, or once a timed-out call's coroutine is cancelled.
    """
    global _sync_timeout, _stream_join_timeout
    if sync_timeout is not None:
        if sync_timeout <= 0:
            raise ValueError(f"sync_timeout must be positive, got {sync_timeout}")
        _sync_timeout = sync_timeout
    if stream_join_timeout is not None:
        if stream_join_timeout <= 0:
            raise ValueError(f"stream_join_timeout must be positive, got {stream_join_timeout}")
        _stream_join_timeout = stream_join_timeout


class _TaskHandle:
    """Cross-thread handle on the task a worker thread's event loop is running.

    The task attaches itself; any thread may then cancel it. A cancel that comes first makes the
    attach fail, so work that has not started never starts.
    """

    __slots__ = ("_cancelled", "_lock", "_loop", "_task")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[Any] | None = None

    def attach(self) -> bool:
        """Bind the running task; ``False`` if cancellation was already requested."""
        with self._lock:
            if self._cancelled:
                return False
            self._loop, self._task = asyncio.get_running_loop(), asyncio.current_task()
            return True

    def detach(self) -> None:
        """Called by the task itself: cancel requests from now on leave it running."""
        with self._lock:
            self._task = None

    def cancel(self) -> None:
        """Cancel the attached task from any thread, or stop it from starting."""
        with self._lock:
            self._cancelled = True
            loop = self._loop
        if loop is not None:
            with contextlib.suppress(RuntimeError):  # the loop already closed: nothing to cancel
                loop.call_soon_threadsafe(self._cancel_attached)

    def _cancel_attached(self) -> None:
        # Runs on the task's own loop, so it is ordered against detach() without races.
        with self._lock:
            task = self._task
        if task is not None:
            task.cancel()


def _run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously.

    Tries ``asyncio.run()`` first. Falls back to a background thread with its
    own event loop when a loop is already running (Jupyter, FastAPI, etc.).
    That fallback is bounded by the sync timeout: on expiry the coroutine is
    cancelled, its thread gets up to the stream join timeout to wind down, and
    ``TimeoutError`` is raised.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running — safe to use asyncio.run().
        return asyncio.run(coro)

    # Loop already running — run in a separate thread. A fresh thread starts with an empty
    # context, so copy the caller's and run inside it: ambient state (e.g. the metering scope's
    # ContextVar) must reach the coroutine.
    result: T | None = None
    exc: BaseException | None = None
    ctx = contextvars.copy_context()
    handle = _TaskHandle()

    async def _main() -> T:
        if not handle.attach():
            coro.close()  # timed out before the worker's loop started
            raise asyncio.CancelledError
        return await coro

    def _target() -> None:
        nonlocal result, exc
        try:
            result = ctx.run(asyncio.run, _main())
        except BaseException as e:
            exc = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=_sync_timeout)
    if thread.is_alive():
        # Cancel rather than abandon: left running, the coroutine would still finish its model call
        # or side-effecting tool, and settle its meter, after the caller has moved on.
        handle.cancel()
        thread.join(timeout=_stream_join_timeout)
        if thread.is_alive():
            logger.warning(
                "Sync wrapper thread still alive %ss after cancelling its timed-out coroutine",
                _stream_join_timeout,
            )
        raise TimeoutError(f"Sync wrapper timed out after {_sync_timeout}s")
    if exc is not None:
        raise exc
    return result  # type: ignore[return-value]


class _StreamBridge[T]:
    """Bounded producer/consumer bridge with cancellable transport ownership."""

    def __init__(self, factory: Callable[[], AsyncIterator[T]]) -> None:
        self.factory = factory
        self.queue: Queue[object] = Queue(maxsize=_STREAM_QUEUE_MAXSIZE)
        self.stop = threading.Event()
        self.handle = _TaskHandle()
        self.context = contextvars.copy_context()
        self.thread = threading.Thread(target=self._target, daemon=True)

    async def _put(self, item: object) -> bool:
        while not self.stop.is_set():
            try:
                self.queue.put_nowait(item)
            except Full:
                await asyncio.sleep(_STREAM_PUT_INTERVAL)
            else:
                return True
        return False

    async def _drain(self) -> None:
        if not self.handle.attach():
            return
        iterator = self.factory()
        try:
            async for item in iterator:
                if not await self._put(item):
                    break
        finally:
            self.handle.detach()
            await close_async(iterator)
        await self._put(_SENTINEL)

    def _target(self) -> None:
        try:
            self.context.run(asyncio.run, self._drain())
        except BaseException as error:
            failure = _SourceError(error)
            while not self.stop.is_set():
                with contextlib.suppress(Full):
                    self.queue.put(failure, timeout=_STREAM_PUT_INTERVAL)
                    return

    def _finish(self, drained: bool) -> None:
        self.stop.set()
        if not drained:
            self.handle.cancel()
        if self.thread is threading.current_thread():
            return
        self.thread.join(timeout=_stream_join_timeout)
        if self.thread.is_alive():
            logger.warning(
                "Stream thread still alive after %ss join timeout", _stream_join_timeout
            )

    def items(self) -> Iterator[T]:
        self.thread.start()
        drained = False
        try:
            while True:
                item = self.queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, _SourceError):
                    raise item.error
                yield cast("T", item)
            drained = True
        finally:
            self._finish(drained)


def _stream_sync[T](async_iterator_factory: Callable[[], AsyncIterator[T]]) -> Iterator[T]:
    """Bridge an async source to a bounded sync iterator, preserving the caller's context.

    A consumer leaving early cancels the drain and closes its source, even while its next item
    is pending. Cleanup cannot be interrupted once it starts. The worker is joined for up to
    the configured stream join timeout, unless it is itself the thread closing the iterator.
    """
    yield from _StreamBridge(async_iterator_factory).items()
