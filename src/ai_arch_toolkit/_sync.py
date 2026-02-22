"""Safe sync wrappers for async code."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from queue import Queue
from typing import TypeVar

T = TypeVar("T")

_SENTINEL = object()


def _run_sync(coro: object) -> T:  # type: ignore[type-var]
    """Run an async coroutine synchronously.

    Tries ``asyncio.run()`` first. Falls back to a background thread with its
    own event loop when a loop is already running (Jupyter, FastAPI, etc.).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running — safe to use asyncio.run().
        return asyncio.run(coro)  # type: ignore[arg-type]

    # Loop already running — run in a separate thread.
    result: T | None = None
    exc: BaseException | None = None

    def _target() -> None:
        nonlocal result, exc
        try:
            result = asyncio.run(coro)  # type: ignore[arg-type]
        except BaseException as e:
            exc = e

    thread = threading.Thread(target=_target)
    thread.start()
    thread.join(timeout=300)
    if thread.is_alive():
        raise TimeoutError("Sync wrapper timed out after 300s")
    if exc is not None:
        raise exc
    return result  # type: ignore[return-value]


def _stream_sync(async_iterator_factory: object) -> Iterator[T]:  # type: ignore[type-var]
    """Bridge an async iterator to a sync one via a thread + queue.

    ``async_iterator_factory`` is a zero-arg callable that returns an
    ``AsyncIterator[T]``.  It is invoked inside the background thread's
    event loop.
    """
    q: Queue[object] = Queue()

    async def _drain() -> None:
        async for item in async_iterator_factory():  # type: ignore[operator]
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
        yield item  # type: ignore[misc]

    thread.join(timeout=5)
