"""A channel for the stream events of the LLM calls made where it is bound (D54).

While a sink is bound, ``LLM.complete`` runs on the stream path and hands every
``StreamEvent`` to the sink as it arrives, with an id of its call; the ``Response`` it returns
is the same. A caller that wants the model's output as it is made (a flow being iterated) binds
one around code that only calls ``complete``. A call made where none is bound does not stream.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Callable, Iterator
from contextvars import ContextVar

from ai_arch_toolkit.core._response import StreamEvent

type LLMEventSink = Callable[[StreamEvent, str], None]
"""Takes one event of a call and the call's id, which every event of that call shares."""

_SINK: ContextVar[LLMEventSink | None] = ContextVar("llm_event_sink", default=None)


def llm_event_sink() -> LLMEventSink | None:
    """The sink bound where this runs, if any."""
    return _SINK.get()


@contextlib.contextmanager
def llm_events_to(sink: LLMEventSink | None) -> Iterator[None]:
    """Bind ``sink`` for the code inside: its ``LLM.complete`` calls stream into it.

    ``None`` unbinds, so the code inside makes plain calls. The binding follows the context, into
    the tasks and threads (``asyncio.to_thread``) started inside it.
    """
    token = _SINK.set(sink)
    try:
        yield
    finally:
        _SINK.reset(token)


def new_call_id() -> str:
    """The id of one streamed call, shared by all its events."""
    return uuid.uuid4().hex[:12]
