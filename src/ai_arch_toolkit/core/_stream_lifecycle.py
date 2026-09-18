"""Explicit stream ownership and optional iterator resource protocols."""

from __future__ import annotations

import contextlib
from typing import Protocol, runtime_checkable


class StreamLifecycle(Protocol):
    """Own a stream's attempt independently of the thread draining its transport."""

    def abandon(self) -> None:
        """Terminate outstanding work without treating partial usage as a success."""
        ...


@runtime_checkable
class AsyncCloseable(Protocol):
    async def aclose(self) -> None: ...


@runtime_checkable
class Closeable(Protocol):
    def close(self) -> None: ...


async def close_async(iterator: object) -> None:
    """Release an iterator's transport without replacing its original outcome."""
    if isinstance(iterator, AsyncCloseable):
        with contextlib.suppress(Exception):
            await iterator.aclose()


def close_sync(iterator: object) -> None:
    if isinstance(iterator, Closeable):
        iterator.close()
