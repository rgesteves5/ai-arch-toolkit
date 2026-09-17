"""xAI adapter failure modes on loopback (gRPC), dummy key."""

from __future__ import annotations

import asyncio
import sys
import time
import traceback
import warnings

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import fakeserver as fs  # noqa: E402
import grpc  # noqa: E402
import xai_sdk  # noqa: E402

from ai_arch_toolkit.core._providers._xai import XAIProvider  # noqa: E402

warnings.simplefilter("ignore")
MSGS = [{"role": "user", "content": "hi"}]


def make(port: int, *, timeout: float = 1.5, retries: int = 0) -> XAIProvider:
    p = XAIProvider("grok-4", "dummy-key")
    p._client = xai_sdk.AsyncClient(
        api_key="dummy-key", api_host=f"localhost:{port}", use_insecure_channel=True,
        timeout=timeout, channel_options=[("grpc.enable_retries", retries)],
    )
    return p


def show_rpc(exc: BaseException) -> None:
    fs.describe(exc)
    cause = exc.__cause__
    if isinstance(cause, grpc.aio.AioRpcError):
        print(f"  grpc code={cause.code()} details={cause.details()!r}")
        print(f"  debug_error_string={cause.debug_error_string()[:260]!r}")


async def run(title: str, port: int, *, stream: bool = False, **kw) -> None:
    print(f"\n=== xai.{'stream' if stream else 'complete'}: {title}")
    p = make(port)
    t0 = time.monotonic()
    try:
        if stream:
            it, state = p.stream(MSGS, max_tokens=16, **kw)
            print("  generator created (no RPC yet)")
            async for _ in it:
                pass
        else:
            await p.complete(MSGS, max_tokens=16, **kw)
        print("  no exception")
    except BaseException as exc:  # noqa: BLE001
        show_rpc(exc)
    print(f"  elapsed={time.monotonic() - t0:.2f}s")


async def main() -> None:
    print("AioRpcError MRO:", [f"{c.__module__}.{c.__qualname__}" for c in grpc.aio.AioRpcError.__mro__])
    print("issubclass(AioRpcError, OSError) =", issubclass(grpc.aio.AioRpcError, OSError))
    await run("closed port", fs.free_port())
    await run("closed port", fs.free_port(), stream=True)
    srv, port, st = await fs.start("hang")  # accepts TCP, never speaks HTTP/2
    await run("TCP accepted, server silent (client timeout=1.5s)", port)
    print(f"  server connections={st.connections}")
    srv.close()
    srv, port, st = await fs.start("reset_on_accept")
    await run("TCP RST on accept", port)
    print(f"  server connections={st.connections}")
    srv.close()
    # SDK-local validation inside chat.create(), which the adapter calls inside its `try` (send region)
    await run("agent_count=3 -> SDK-local validation in chat.create()", fs.free_port(), agent_count=3)
    await run("agent_count=3, stream", fs.free_port(), stream=True, agent_count=3)
    # adapter-local NotImplementedError for server tools (build)
    print("\n=== xai.complete: server tool -> adapter NotImplementedError (build)")
    p = make(fs.free_port())
    try:
        await p.complete(MSGS, tools=[{"_server_tool": True, "type": "web_search", "name": "web_search"}])
    except BaseException as exc:  # noqa: BLE001
        fs.describe(exc)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        traceback.print_exc()
