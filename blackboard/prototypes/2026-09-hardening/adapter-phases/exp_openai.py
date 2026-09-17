"""OpenAI + Meta adapter failure modes against a loopback fake server (dummy key)."""

from __future__ import annotations

import asyncio
import sys
import traceback
import warnings

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import fakeserver as fs  # noqa: E402

from ai_arch_toolkit.core._providers._meta import MetaProvider  # noqa: E402
from ai_arch_toolkit.core._providers._openai import OpenAIProvider  # noqa: E402

warnings.simplefilter("ignore")
MSGS = [{"role": "user", "content": "hi"}]


def chunk(content: str | None = None, finish: str | None = None) -> dict:
    return {
        "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": finish}],
    }


def oai(port: int) -> OpenAIProvider:
    return OpenAIProvider("gpt-4o-mini", "dummy-key", base_url=f"http://127.0.0.1:{port}/v1", timeout=1.0)


def meta(port: int) -> MetaProvider:
    return MetaProvider("muse-spark-1", "dummy-key", base_url=f"http://127.0.0.1:{port}/v1", timeout=1.0)


async def run_complete(make, title: str, port: int, **kw) -> None:
    print(f"\n=== {make.__name__}.complete: {title}")
    p = make(port)
    try:
        r = await p.complete(MSGS, max_tokens=16, **kw)
        print("  no exception ->", repr(r)[:160])
    except BaseException as exc:  # noqa: BLE001
        fs.describe(exc)
    finally:
        await p.close()


async def run_stream(make, title: str, port: int) -> None:
    print(f"\n=== {make.__name__}.stream_events: {title}")
    p = make(port)
    got: list = []
    state = None
    try:
        it, state = p.stream_events(MSGS, max_tokens=16)
        async for item in it:
            got.append((item.kind, item.text))
        print("  no exception; items:", got)
    except BaseException as exc:  # noqa: BLE001
        print(f"  items before failure: {got!r}")
        fs.describe(exc)
    finally:
        if state is not None:
            print(f"  state.usage at end: {state.usage!r}")
        await p.close()


async def main() -> None:
    for make in (oai, meta):
        await run_complete(make, "closed port", fs.free_port())
        srv, port, st = await fs.start("hang")
        await run_complete(make, "server hangs after request (timeout=1s)", port)
        print(f"  server saw requests={st.requests} paths={st.paths}")
        srv.close()
        srv, port, st = await fs.start("disconnect")
        await run_complete(make, "server closes without response", port)
        srv.close()
        for status, headers in ((429, {"retry-after": "7"}), (503, {}), (400, {})):
            srv, port, st = await fs.start(
                "status", status=status, headers=headers,
                body={"error": {"message": "x", "type": "t", "code": "c"}},
            )
            await run_complete(make, f"HTTP {status}", port)
            srv.close()
        srv, port, st = await fs.start("status", status=200, body={"unexpected": True})
        await run_complete(make, "HTTP 200 with an unexpected JSON body (parse)", port)
        srv.close()

    # Meta: HTTP 200 whose response.status == "failed"
    failed = {"id": "r1", "object": "response", "status": "failed", "model": "muse-spark-1", "output": [],
              "error": {"code": "server_error", "message": "boom"},
              "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}}
    srv, port, st = await fs.start("status", status=200, body=failed)
    await run_complete(meta, "HTTP 200 + response.status=='failed' (usage present in body)", port)
    srv.close()

    # stream timing
    print("\n=== oai.stream_events: request issued at creation or first __anext__?")
    srv, port, st = await fs.start("sse", events=[fs.sse(None, chunk("Hel")), fs.sse(None, "[DONE]")])
    p = oai(port)
    it, state = p.stream_events(MSGS, max_tokens=16)
    await asyncio.sleep(0.2)
    print(f"  after stream_events(): connections={st.connections} requests={st.requests}")
    await it.__anext__()
    print(f"  after first __anext__: connections={st.connections} requests={st.requests}")
    await it.aclose()
    await p.close()
    srv.close()

    # OpenAI mid-stream error payload
    err = {"error": {"message": "boom", "type": "server_error", "code": "server_error"}}
    srv, port, st = await fs.start("sse", events=[fs.sse(None, chunk("Hel")), fs.sse(None, err)])
    await run_stream(oai, "mid-stream `data: {error: ...}` after a delta", port)
    srv.close()
    srv, port, st = await fs.start("sse", events=[fs.sse("error", err)])
    await run_stream(oai, "`event: error` as the first event", port)
    srv.close()
    for end in ("drop", "hang"):
        srv, port, st = await fs.start("sse", events=[fs.sse(None, chunk("Hel"))], end=end, seconds=5)
        await run_stream(oai, f"mid-stream transport failure: {end}", port)
        srv.close()

    # Meta mid-stream
    delta = {"type": "response.output_text.delta", "item_id": "m1", "output_index": 0,
             "content_index": 0, "delta": "Hel", "sequence_number": 1, "logprobs": []}
    srv, port, st = await fs.start(
        "sse", events=[fs.sse("response.output_text.delta", delta),
                       fs.sse("response.failed", {"type": "response.failed", "sequence_number": 2, "response": failed})])
    await run_stream(meta, "mid-stream response.failed (usage present in event)", port)
    srv.close()
    srv, port, st = await fs.start(
        "sse", events=[fs.sse("response.output_text.delta", delta),
                       fs.sse("error", {"type": "error", "code": "service_overloaded", "message": "busy",
                                        "param": None, "sequence_number": 2})])
    await run_stream(meta, "mid-stream `error` event (flat shape)", port)
    srv.close()
    srv, port, st = await fs.start(
        "sse", events=[fs.sse("response.output_text.delta", delta), fs.sse("error", err)])
    await run_stream(meta, "mid-stream `error` event (nested {error:{}} shape -> SDK raises APIError)", port)
    srv.close()
    srv, port, st = await fs.start("sse", events=[fs.sse("response.output_text.delta", delta)], end="drop")
    await run_stream(meta, "mid-stream transport failure: drop", port)
    srv.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        traceback.print_exc()
