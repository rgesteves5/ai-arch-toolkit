"""Meta: replay after stream_events() with two parallel calls (fake SDK events from the unit tests)."""
from __future__ import annotations

import asyncio
import json

from ai_arch_toolkit.core import LLM
from ai_arch_toolkit.core._content import tool_result
from ai_arch_toolkit.core._providers._meta import _input_items
from tests import test_meta_provider as T


async def main() -> None:
    cls = T.TestStream if hasattr(T, "TestStream") else None
    holder = next(
        v for v in vars(T).values() if isinstance(v, type) and hasattr(v, "TOOL_EVENTS") and hasattr(v, "_final")
    )
    inst = holder()
    client = T._events(*holder.TOOL_EVENTS, T._completed(inst._final()))
    async with LLM(T.MODEL, api_key="test-key") as llm:
        llm._provider._client = client
        stream = llm.stream_events([T.USER], tools=[T.WEATHER_TOOL])
        arrival = [e.tool_call.id async for e in stream if e.kind == "tool_call"]
        turn = stream.response
    print("tool_call events arrival order:", arrival, "| turn.tool_calls order:", [tc.id for tc in turn.tool_calls])
    hist = [T.USER, turn.to_message()] + [
        tool_result("ok-" + tc.id, tool_use_id=tc.id, name=tc.name) for tc in turn.tool_calls
    ]
    for it in _input_items(hist):
        d = dict(it)
        if "encrypted_content" in d:
            d["encrypted_content"] = d["encrypted_content"][:12] + "..."
        print("  ", json.dumps(d)[:200])


asyncio.run(main())
