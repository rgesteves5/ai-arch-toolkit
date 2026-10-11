"""complete_sync inside a running loop, after an async call on the same LLM (loopback only)."""

from __future__ import annotations

import pathlib
import asyncio
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))  # the repo root, 4 levels up

from ai_arch_toolkit import LLM
from ai_arch_toolkit.core import _sync
from tests.integration import fakeserver
from tests.integration.test_clients_across_loops import COMPATIBLE

_sync.configure_sync_timeouts(sync_timeout=5, stream_join_timeout=1)


async def main(port: int) -> None:
    llm = LLM("llama3.2", base_url=f"http://127.0.0.1:{port}/v1", api_key="local-test", max_tokens=16)
    print("async:", (await llm.complete("hi")).text)  # client built on this loop, pooled conn
    t0 = time.monotonic()
    try:
        print("sync in loop:", llm.complete_sync("hi").text)  # a worker thread, another loop
    except BaseException as exc:  # noqa: BLE001
        print("sync in loop raised:", type(exc).__name__, str(exc)[:200])
    print(f"took {time.monotonic() - t0:.2f}s")


with fakeserver.KeepAlive(COMPATIBLE) as server:
    asyncio.run(main(server.port))
    print("server requests:", server.stats.requests)
