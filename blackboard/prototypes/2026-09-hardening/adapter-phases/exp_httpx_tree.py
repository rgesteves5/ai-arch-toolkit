"""httpx exception tree + how anthropic/openai wrap each transport failure (custom transport, no I/O)."""

from __future__ import annotations

import asyncio
import inspect

import anthropic
import httpx
import openai


def tree(cls: type, depth: int = 0) -> None:
    print("  " * depth + cls.__name__)
    for sub in sorted(cls.__subclasses__(), key=lambda c: c.__name__):
        if sub.__module__.startswith("httpx"):
            tree(sub, depth + 1)


print("httpx", httpx.__version__, "| anthropic imports:", inspect.getmodule(anthropic.AsyncAnthropic).__name__)
print("anthropic._base_client httpx module is httpx:", anthropic._base_client.httpx is httpx)
print("openai._base_client httpx module is httpx:", openai._base_client.httpx is httpx)
tree(httpx.HTTPError)
print("issubclass(httpx.TransportError, OSError) =", issubclass(httpx.TransportError, OSError))
print("issubclass(httpx.TimeoutException, TimeoutError) =", issubclass(httpx.TimeoutException, TimeoutError))

CASES = [
    httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.UnsupportedProtocol,
    httpx.LocalProtocolError, httpx.ProxyError, httpx.WriteError, httpx.WriteTimeout,
    httpx.ReadError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.CloseError,
]


class Raising(httpx.AsyncBaseTransport):
    def __init__(self, exc_type: type[Exception]) -> None:
        self.exc_type = exc_type

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise self.exc_type("simulated", request=request)


async def main() -> None:
    print(f"\n{'httpx raised':22s} {'anthropic wraps as':28s} {'openai wraps as':26s} __cause__ preserved")
    for exc_type in CASES:
        row = [exc_type.__name__]
        ok = []
        for sdk, make in (
            (anthropic, lambda c: anthropic.AsyncAnthropic(api_key="dummy", max_retries=0, http_client=c)),
            (openai, lambda c: openai.AsyncOpenAI(api_key="dummy", max_retries=0, http_client=c)),
        ):
            client = make(httpx.AsyncClient(transport=Raising(exc_type)))
            try:
                if sdk is anthropic:
                    await client.messages.create(model="m", max_tokens=8, messages=[{"role": "user", "content": "x"}])
                else:
                    await client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
            except Exception as exc:  # noqa: BLE001
                row.append(type(exc).__name__)
                ok.append(type(exc.__cause__) is exc_type)
            await client.close()
        print(f"{row[0]:22s} {row[1]:28s} {row[2]:26s} {ok}")


asyncio.run(main())
