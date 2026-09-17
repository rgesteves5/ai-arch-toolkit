"""Gemini adapter failure modes against a loopback fake server (dummy key), aiohttp and httpx stacks."""

from __future__ import annotations

import asyncio
import random
import sys
import time
import traceback
import warnings

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import fakeserver as fs  # noqa: E402
import httpx  # noqa: E402
from google import genai  # noqa: E402

from ai_arch_toolkit.core._providers._gemini import GeminiProvider  # noqa: E402

warnings.simplefilter("ignore")
random.randint = lambda a, b: 0  # SDK sleeps 1 + randint(0, 9) s before its hidden aiohttp re-send
MSGS = [{"role": "user", "content": "hi"}]
OK_CHUNK = {"candidates": [{"content": {"role": "model", "parts": [{"text": "Hel"}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1, "totalTokenCount": 6}}


def make(port: int, stack: str) -> GeminiProvider:
    p = GeminiProvider("gemini-2.5-flash", "dummy-key", timeout=1.0)
    opts: dict = {"base_url": f"http://127.0.0.1:{port}", "retry_options": {"attempts": 1}, "timeout": 1000}
    if stack == "httpx":
        opts["httpx_async_client"] = httpx.AsyncClient()
    client = genai.Client(api_key="dummy-key", http_options=opts)
    p._client = client
    print(f"  [stack={stack}] SDK _use_aiohttp() = {client._api_client._use_aiohttp()}")
    return p


async def run_complete(stack: str, title: str, port: int, st: fs.Stats | None = None) -> None:
    print(f"\n=== gemini.complete [{stack}]: {title}")
    p = make(port, stack)
    t0 = time.monotonic()
    try:
        r = await p.complete(MSGS, max_tokens=16)
        print("  no exception ->", repr(r)[:140])
    except BaseException as exc:  # noqa: BLE001
        fs.describe(exc)
    print(f"  elapsed={time.monotonic() - t0:.2f}s" + (f" server connections={st.connections} requests={st.requests}" if st else ""))


async def run_stream(stack: str, title: str, port: int, st: fs.Stats | None = None) -> None:
    print(f"\n=== gemini.stream [{stack}]: {title}")
    p = make(port, stack)
    got: list = []
    state = None
    try:
        it, state = p.stream(MSGS, max_tokens=16)
        await asyncio.sleep(0.2)
        if st is not None:
            print(f"  after provider.stream(): connections={st.connections} requests={st.requests}")
        async for item in it:
            got.append(item)
        print("  no exception; items:", got)
    except BaseException as exc:  # noqa: BLE001
        print(f"  items before failure: {got!r}")
        fs.describe(exc)
    if state is not None:
        print(f"  state.usage at end: {state.usage!r}")
    if st is not None:
        print(f"  server connections={st.connections} requests={st.requests}")


async def main() -> None:
    for stack in ("aiohttp", "httpx"):
        await run_complete(stack, "closed port", fs.free_port())
        srv, port, st = await fs.start("hang")
        await run_complete(stack, "server hangs after request (timeout=1s)", port, st)
        srv.close()
        srv, port, st = await fs.start("disconnect")
        await run_complete(stack, "server closes without response", port, st)
        srv.close()
        for status, headers in ((429, {"retry-after": "7"}), (503, {}), (400, {})):
            body = {"error": {"code": status, "message": "x", "status": "S"}}
            srv, port, st = await fs.start("status", status=status, headers=headers, body=body)
            await run_complete(stack, f"HTTP {status}", port, st)
            srv.close()
        srv, port, st = await fs.start("status", status=200, body={"unexpected": True})
        await run_complete(stack, "HTTP 200 with an unexpected JSON body (parse)", port, st)
        srv.close()
        srv, port, st = await fs.start("sse", events=[fs.sse(None, OK_CHUNK)])
        await run_stream(stack, "normal one-chunk stream (request timing)", port, st)
        srv.close()
        err = {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}
        srv, port, st = await fs.start("sse", events=[fs.sse(None, OK_CHUNK), fs.sse(None, err)])
        await run_stream(stack, "mid-stream `data: {error}` chunk after a text chunk", port, st)
        srv.close()
        srv, port, st = await fs.start("sse", events=[fs.sse(None, OK_CHUNK)], end="drop")
        await run_stream(stack, "mid-stream transport failure: drop", port, st)
        srv.close()
        srv, port, st = await fs.start("sse", events=[fs.sse(None, OK_CHUNK)], end="hang", seconds=4)
        await run_stream(stack, "mid-stream transport failure: hang (timeout=1s)", port, st)
        srv.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        traceback.print_exc()
