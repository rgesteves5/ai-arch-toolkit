"""Loopback HTTP server that drives the official SDKs without leaving 127.0.0.1."""

from __future__ import annotations

import asyncio
import json
import socket
import struct
import threading
from dataclasses import dataclass, field
from typing import Literal

type Behaviour = Literal["status", "hang", "close", "reset", "sse", "raw", "script"]
type Ending = Literal["finish", "drop", "reset", "hang"]


@dataclass(slots=True)
class Stats:
    """Requests the server has read."""

    requests: int = 0
    bodies: list[bytes] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Reply:
    """One answer of a ``script``: a JSON body with a 200, or server-sent events that finish."""

    body: dict[str, object] | None = None
    events: list[bytes] | None = None


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


async def _answer(
    writer: asyncio.StreamWriter, status: int, payload: bytes, headers: dict[str, str] | None
) -> None:
    extra = "".join(f"{k}: {v}\r\n" for k, v in (headers or {}).items())
    head = (
        f"HTTP/1.1 {status} X\r\ncontent-type: application/json\r\n{extra}"
        f"content-length: {len(payload)}\r\n\r\n"
    )
    writer.write(head.encode() + payload)
    await writer.drain()


async def _reply(writer: asyncio.StreamWriter, reply: Reply, seconds: float) -> None:
    if reply.events is not None:
        await _stream(writer, reply.events, "finish", seconds)
    else:
        await _answer(writer, 200, json.dumps(reply.body or {}).encode(), None)


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
    replies: list[Reply] | None = None,
) -> tuple[asyncio.Server, int, Stats]:
    """Serve one behaviour on an ephemeral port.

    ``status`` answers with ``status`` and ``body`` as JSON (``raw``: with ``raw`` bytes);
    ``hang`` reads the request and sends nothing for ``seconds``; ``close`` closes without an
    answer; ``reset`` aborts with a TCP RST; ``sse`` streams ``events`` and then ends as
    ``ending`` says; ``script`` answers the n-th request with ``replies[n]``, for a conversation
    of several turns.
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
            elif behaviour == "script":
                await _reply(writer, (replies or [])[stats.requests - 1], seconds)
            elif behaviour in ("status", "raw"):
                payload = raw if behaviour == "raw" else json.dumps(body or {}).encode()
                await _answer(writer, status, payload, headers)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1], stats


class KeepAlive:
    """A loopback HTTP server in a thread of its own that keeps each connection open across
    requests, as a real API does: a sync call's pooled connection outlives the call's event loop.

    Every request gets ``body`` as JSON with a 200. Use it as a context manager.
    """

    def __init__(self, body: dict[str, object]) -> None:
        self.body = body
        self.stats = Stats()
        self.port = 0
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._thread = threading.Thread(target=self._run, name="keep-alive", daemon=True)

    def __enter__(self) -> KeepAlive:
        self._thread.start()
        self._ready.wait()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join()

    def _run(self) -> None:
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = server.sockets[0].getsockname()[1]
        self._ready.set()
        await self._stop.wait()
        # Not ``wait_closed()``: a client may leave its connection open (one whose loop died),
        # and asyncio.run cancels the handlers that still read from one.
        server.close()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                self.stats.bodies.append(await _read_body(reader))
                self.stats.requests += 1
                payload = json.dumps(self.body).encode()
                head = (
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
                    f"content-length: {len(payload)}\r\n\r\n"
                )
                writer.write(head.encode() + payload)
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()
