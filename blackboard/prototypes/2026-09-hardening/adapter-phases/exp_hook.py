"""(1) Do the SDKs expose Default httpx clients accepting event_hooks? (2) When does the request
hook fire? (3) What escapes the SDK call for a 200 whose body is not JSON? Loopback only."""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import anthropic  # noqa: E402
import fakeserver as fs  # noqa: E402
import httpx  # noqa: E402
import openai  # noqa: E402

print("anthropic.DefaultAsyncHttpxClient:", anthropic.DefaultAsyncHttpxClient.__mro__[1].__name__)
print("openai.DefaultAsyncHttpxClient:   ", openai.DefaultAsyncHttpxClient.__mro__[1].__name__)


async def raw_200(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    await fs._read_request(reader)
    body = b"<html>not json</html>"
    writer.write(
        b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: "
        + str(len(body)).encode() + b"\r\n\r\n" + body
    )
    await writer.drain()
    writer.close()


async def main() -> None:
    fired: list[str] = []

    async def on_request(request: httpx.Request) -> None:
        fired.append(str(request.url))

    # (2) hook timing: closed port, and SDK-local error
    closed = fs.free_port()
    for title, kwargs in (
        ("closed port", {"max_tokens": 8}),
        ("SDK-local ValueError (max_tokens=500k, default timeout)", {"max_tokens": 500_000}),
    ):
        fired.clear()
        client = anthropic.AsyncAnthropic(
            api_key="dummy", base_url=f"http://127.0.0.1:{closed}", max_retries=0,
            http_client=anthropic.DefaultAsyncHttpxClient(event_hooks={"request": [on_request]}),
        )
        try:
            await client.messages.create(model="m", messages=[{"role": "user", "content": "x"}], **kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"anthropic {title}: {type(exc).__name__}; request hook fired={bool(fired)}")
        await client.close()

    # (3) 200 + non-JSON body
    port = fs.free_port()
    srv = await asyncio.start_server(raw_200, "127.0.0.1", port)
    for name, call in (
        ("anthropic", lambda: anthropic.AsyncAnthropic(api_key="dummy", base_url=f"http://127.0.0.1:{port}", max_retries=0)
            .messages.create(model="m", max_tokens=8, messages=[{"role": "user", "content": "x"}])),
        ("openai", lambda: openai.AsyncOpenAI(api_key="dummy", base_url=f"http://127.0.0.1:{port}/v1", max_retries=0)
            .chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])),
    ):
        try:
            r = await call()
            print(f"{name} 200 non-JSON body: no exception, returned {type(r).__name__}: {str(r)[:60]!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"{name} 200 non-JSON body: {type(exc).__module__}.{type(exc).__name__} "
                  f"mro={[c.__name__ for c in type(exc).__mro__[1:4]]}")
    srv.close()


asyncio.run(main())
