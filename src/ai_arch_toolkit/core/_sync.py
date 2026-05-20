"""Safe sync wrappers for async code."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from queue import Queue
from typing import Any, cast

logger = logging.getLogger(__name__)

# Configurable timeout defaults (seconds) — overridable via env vars or configure_sync_timeouts()
_sync_timeout: float = float(os.environ.get("AI_ARCH_SYNC_TIMEOUT", "300"))
_stream_join_timeout: float = float(os.environ.get("AI_ARCH_STREAM_JOIN_TIMEOUT", "5"))

_SENTINEL = object()


def configure_sync_timeouts(
    sync_timeout: float | None = None,
    stream_join_timeout: float | None = None,
) -> None:
    """Configure sync wrapper timeouts."""
    global _sync_timeout, _stream_join_timeout
    if sync_timeout is not None:
        if sync_timeout <= 0:
            raise ValueError(f"sync_timeout must be positive, got {sync_timeout}")
        _sync_timeout = sync_timeout
    if stream_join_timeout is not None:
        if stream_join_timeout <= 0:
            raise ValueError(f"stream_join_timeout must be positive, got {stream_join_timeout}")
        _stream_join_timeout = stream_join_timeout


def _run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously.

    Tries ``asyncio.run()`` first. Falls back to a background thread with its
    own event loop when a loop is already running (Jupyter, FastAPI, etc.).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running — safe to use asyncio.run().
        return asyncio.run(coro)

    # Loop already running — run in a separate thread.
    result: T | None = None
    exc: BaseException | None = None

    def _target() -> None:
        nonlocal result, exc
        try:
            result = asyncio.run(coro)  # type: ignore[arg-type]
        except BaseException as e:
            exc = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=_sync_timeout)
    if thread.is_alive():
        raise TimeoutError(f"Sync wrapper timed out after {_sync_timeout}s")
    if exc is not None:
        raise exc
    return result  # type: ignore[return-value]


def _stream_sync[T](async_iterator_factory: Callable[[], AsyncIterator[T]]) -> Iterator[T]:
    """Bridge an async iterator to a sync one via a thread + queue.

    ``async_iterator_factory`` is a zero-arg callable that returns an
    ``AsyncIterator[T]``.  It is invoked inside the background thread's
    event loop.
    """
    q: Queue[object] = Queue()

    async def _drain() -> None:
        async for item in async_iterator_factory():
            q.put(item)
        q.put(_SENTINEL)

    def _target() -> None:
        try:
            asyncio.run(_drain())
        except BaseException as e:
            q.put(e)
            q.put(_SENTINEL)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    while True:
        item = q.get()
        if item is _SENTINEL:
            break
        if isinstance(item, BaseException):
            raise item
        yield cast(T, item)

    thread.join(timeout=_stream_join_timeout)
    if thread.is_alive():
        logger.warning("Stream thread still alive after %ss join timeout", _stream_join_timeout)
