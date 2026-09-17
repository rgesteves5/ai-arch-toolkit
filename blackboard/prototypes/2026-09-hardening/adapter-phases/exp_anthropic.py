"""Anthropic adapter failure modes against a loopback fake server (dummy key, no external I/O)."""

from __future__ import annotations

import asyncio
import sys
import traceback
import warnings

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import fakeserver as fs  # noqa: E402

from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider  # noqa: E402

warnings.simplefilter("ignore")
MSGS = [{"role": "user", "content": "hi"}]

MESSAGE_START = {
    "type": "message_start",
    "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 25, "output_tokens": 1},
    },
}
BLOCK_START = {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
DELTA = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}
ERROR = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}


def provider(port: int, timeout: float = 1.0) -> AnthropicProvider:
    return AnthropicProvider(
        "claude-sonnet-4-5", "dummy-key", base_url=f"http://127.0.0.1:{port}", timeout=timeout
    )


async def run_complete(title: str, port: int, **kw) -> None:
    print(f"\n=== complete: {title}")
    p = provider(port)
    try:
        await p.complete(MSGS, max_tokens=16, **kw)
        print("  no exception")
    except BaseException as exc:  # noqa: BLE001
        fs.describe(exc)
    finally:
        await p.close()


async def run_stream(title: str, port: int, *, events: bool = False) -> None:
    print(f"\n=== {'stream_events' if events else 'stream'}: {title}")
    p = provider(port)
    got: list = []
    state = None
    try:
        it, state = (p.stream_events if events else p.stream)(MSGS, max_tokens=16)
        print("  generator created; server connections so far = (see below)")
        async for item in it:
            got.append(item)
        print("  no exception; items:", got)
    except BaseException as exc:  # noqa: BLE001
        print(f"  items before failure: {got!r}")
        fs.describe(exc)
    finally:
        if state is not None:
            print(f"  state.usage at end: {state.usage!r}")
        await p.close()


async def main() -> None:
    # 1. never connected
    await run_complete("closed port (connect refused)", fs.free_port())
    # 2. sent, no response (read timeout)
    srv, port, st = await fs.start("hang")
    await run_complete("server hangs after request (read timeout, timeout=1s)", port)
    print(f"  server saw requests={st.requests}")
    srv.close()
    # 3. sent, server closes without response
    srv, port, st = await fs.start("disconnect")
    await run_complete("server closes without response", port)
    print(f"  server saw requests={st.requests}")
    srv.close()
    # 3b. RST after request
    srv, port, st = await fs.start("reset")
    await run_complete("server RSTs after request", port)
    srv.close()
    # 4. 429 / 503 / 529 / 400
    for status, body, headers in (
        (429, {"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}, {"retry-after": "7"}),
        (503, {"type": "error", "error": {"type": "api_error", "message": "unavail"}}, {}),
        (529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}, {}),
        (400, {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}, {}),
    ):
        srv, port, st = await fs.start("status", status=status, body=body, headers=headers)
        await run_complete(f"HTTP {status}", port)
        srv.close()
    # 5. local SDK validation inside create(): huge max_tokens non-streaming, DEFAULT timeout
    print("\n=== complete: SDK-local ValueError (max_tokens too large for non-streaming, default timeout)")
    srv, port, st = await fs.start("hang")
    p = AnthropicProvider("claude-sonnet-4-5", "dummy-key", base_url=f"http://127.0.0.1:{port}")
    try:
        await p.complete(MSGS, max_tokens=500_000)
    except BaseException as exc:  # noqa: BLE001
        fs.describe(exc)
    print(f"  server saw connections={st.connections} requests={st.requests}  (0 => nothing was sent)")
    await p.close()
    srv.close()
    # 6. 200 with a body that is not a Message (parse phase)
    srv, port, st = await fs.start("status", status=200, body={"unexpected": True})
    await run_complete("HTTP 200 with a non-Message JSON body (parse)", port)
    srv.close()
    # 7. streams: when is the request issued?
    print("\n=== stream: is the HTTP request issued at generator creation or first __anext__?")
    srv, port, st = await fs.start("sse", events=[fs.sse("message_start", MESSAGE_START)], end="finish")
    p = provider(port)
    it, state = p.stream(MSGS, max_tokens=16)
    await asyncio.sleep(0.2)
    print(f"  after provider.stream(): connections={st.connections} requests={st.requests}")
    try:
        await it.__anext__()
    except StopAsyncIteration:
        pass
    except BaseException as exc:  # noqa: BLE001
        fs.describe(exc)
    print(f"  after first __anext__:   connections={st.connections} requests={st.requests}")
    await p.close()
    srv.close()
    # 8. mid-stream error SSE event
    evs = [fs.sse("message_start", MESSAGE_START), fs.sse("content_block_start", BLOCK_START),
           fs.sse("content_block_delta", DELTA), fs.sse("error", ERROR)]
    srv, port, st = await fs.start("sse", events=evs, end="finish")
    await run_stream("mid-stream `error` SSE event (overloaded_error) after a text delta", port)
    srv.close()
    srv, port, st = await fs.start("sse", events=evs, end="finish")
    await run_stream("same, stream_events", port, events=True)
    srv.close()
    # 8b. error event BEFORE any delta
    srv, port, st = await fs.start("sse", events=[fs.sse("error", ERROR)], end="finish")
    await run_stream("`error` SSE event as the first event", port)
    srv.close()
    # 9. mid-stream connection drop / reset / stall
    for end in ("drop", "reset", "hang"):
        srv, port, st = await fs.start("sse", events=evs[:3], end=end, seconds=5)
        await run_stream(f"mid-stream transport failure: {end}", port)
        srv.close()
    # 10. connect refused on stream
    await run_stream("closed port", fs.free_port())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        traceback.print_exc()
