"""Loopback HTTP server that drives the official SDKs without leaving 127.0.0.1."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class Stats:
    """Requests the server has read."""

    requests: int = 0
    bodies: list[bytes] = field(default_factory=list)


async def _read_body(reader: asyncio.StreamReader) -> bytes:
    head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1")
    length = next(
        (
            int(line.split(":", 1)[1])
            for line in head.split("\r\n")
            if line.lower().startswith("content-length:")
        ),
        0,
    )
    return await reader.readexactly(length) if length else b""


async def start(
    behaviour: Literal["status", "hang"],
    *,
    status: int = 200,
    body: dict[str, object] | None = None,
    seconds: float = 30.0,
) -> tuple[asyncio.Server, int, Stats]:
    """Serve one behaviour on an ephemeral port.

    ``status`` answers every request with ``status`` and ``body`` as JSON; ``hang`` reads the
    request and sends nothing for ``seconds``.
    """
    stats = Stats()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            stats.bodies.append(await _read_body(reader))
            stats.requests += 1
            if behaviour == "hang":
                await asyncio.sleep(seconds)
                return
            payload = json.dumps(body or {}).encode()
            head = (
                f"HTTP/1.1 {status} X\r\ncontent-type: application/json\r\n"
                f"content-length: {len(payload)}\r\n\r\n"
            )
            writer.write(head.encode() + payload)
            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1], stats
