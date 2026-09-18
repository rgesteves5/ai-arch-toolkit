"""Loopback HTTP server that drives the official SDKs without leaving 127.0.0.1."""

from __future__ import annotations

import asyncio
import json
import socket
import struct
from dataclasses import dataclass, field
from typing import Literal

type Behaviour = Literal["status", "hang", "close", "reset", "sse", "raw"]
type Ending = Literal["finish", "drop", "reset", "hang"]


@dataclass(slots=True)
class Stats:
    """Requests the server has read."""

    requests: int = 0
    bodies: list[bytes] = field(default_factory=list)


def sse(data: object, event: str | None = None) -> bytes:
    """One server-sent event; ``data`` is JSON-encoded unless it is already a string."""
    payload = data if isinstance(data, str) else json.dumps(data)
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {payload}\n\n".encode()


def closed_port() -> int:
    """A loopback port with nothing listening (a connection is refused)."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


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


def _reset(writer: asyncio.StreamWriter) -> None:
    """Abort the connection with a TCP RST instead of a FIN."""
    sock = writer.get_extra_info("socket")
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    writer.transport.abort()


def _chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


async def _stream(
    writer: asyncio.StreamWriter, events: list[bytes], ending: Ending, seconds: float
) -> None:
    head = (
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
    )
    writer.write(head.encode())
    for event in events:
        writer.write(_chunk(event))
        await writer.drain()
        await asyncio.sleep(0.01)
    if ending == "finish":
        writer.write(b"0\r\n\r\n")
        await writer.drain()
    elif ending == "reset":
        _reset(writer)
    elif ending == "hang":
        await asyncio.sleep(seconds)
    # "drop": close with a FIN and no terminating chunk


async def start(
    behaviour: Behaviour,
    *,
    status: int = 200,
    body: dict[str, object] | None = None,
    raw: bytes = b"",
    headers: dict[str, str] | None = None,
    events: list[bytes] | None = None,
    ending: Ending = "finish",
    seconds: float = 30.0,
) -> tuple[asyncio.Server, int, Stats]:
    """Serve one behaviour on an ephemeral port.

    ``status`` answers with ``status`` and ``body`` as JSON (``raw``: with ``raw`` bytes);
    ``hang`` reads the request and sends nothing for ``seconds``; ``close`` closes without an
    answer; ``reset`` aborts with a TCP RST; ``sse`` streams ``events`` and then ends as
    ``ending`` says.
    """
    stats = Stats()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            stats.bodies.append(await _read_body(reader))
            stats.requests += 1
            if behaviour == "hang":
                await asyncio.sleep(seconds)
            elif behaviour == "reset":
                _reset(writer)
            elif behaviour == "sse":
                await _stream(writer, events or [], ending, seconds)
            elif behaviour in ("status", "raw"):
                payload = raw if behaviour == "raw" else json.dumps(body or {}).encode()
                extra = "".join(f"{k}: {v}\r\n" for k, v in (headers or {}).items())
                head = (
                    f"HTTP/1.1 {status} X\r\ncontent-type: application/json\r\n{extra}"
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
