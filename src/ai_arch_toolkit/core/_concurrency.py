"""Run-scoped concurrency limiting for LLM inference.

A global ceiling on how many ``LLM.complete()`` calls hit the model server at
once, shared across a whole run — every nested flow, agent, and fallback. This
protects a resource with a hard concurrency limit (a local GPU/CPU, a
rate-limited endpoint, a connection pool) *regardless of how the orchestration is
shaped or nested*. It is model-agnostic: it applies to cloud and local models
alike (default = unlimited, so it is inert until you opt in).

It is deliberately **leaf-level** — the slot is acquired around the single
provider call, never held across nested orchestration — so it cannot deadlock a
flow that fans out into nested flows. Contrast ``Flow(max_parallelism=...)``,
which bounds the *width* of one flow's fan-out (see ``toolkit/flow``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from types import TracebackType

# The ambient inference limiter. ``None`` means "no cap" (the default). A single
# Semaphore instance is shared across every task spawned under the scope, because
# asyncio copies the context (same ContextVar value) into each child task.
_inference_sem: ContextVar[asyncio.Semaphore | None] = ContextVar("inference_sem", default=None)


@contextmanager
def inference_limit(max_concurrent: int) -> Iterator[None]:
    """Cap concurrent LLM inference calls within this scope and every nested one.

    Use it around a run to keep at most ``max_concurrent`` ``complete()`` calls in
    flight at any instant, no matter how many parallel steps, nested agents, or
    fallbacks are active::

        with inference_limit(2):            # e.g. a local GPU that handles 2 at a time
            result = agent.run_sync(task)   # never more than 2 concurrent inferences

    Nesting is allowed; the innermost scope wins (its own, independent semaphore).
    ``max_concurrent`` must be >= 1. Streaming calls are not throttled by this
    limit (they are rarely fanned out); it governs ``complete()`` / ``complete_sync()``.
    """
    if max_concurrent < 1:
        raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")
    token = _inference_sem.set(asyncio.Semaphore(max_concurrent))
    try:
        yield
    finally:
        _inference_sem.reset(token)


class inference_slot:
    """Async context manager: acquire the ambient inference semaphore, or no-op.

    Held only around the actual provider call, so it never spans nested
    orchestration. A no-op when no ``inference_limit`` scope is active.
    """

    __slots__ = ("_sem",)

    def __init__(self) -> None:
        self._sem: asyncio.Semaphore | None = None

    async def __aenter__(self) -> None:
        # Read once at acquire time and hold the same instance for release, so a
        # concurrent scope change can never mismatch acquire/release.
        self._sem = _inference_sem.get()
        if self._sem is not None:
            await self._sem.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._sem is not None:
            self._sem.release()
