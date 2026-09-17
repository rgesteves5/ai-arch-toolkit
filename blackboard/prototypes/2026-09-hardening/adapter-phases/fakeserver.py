"""Loopback-only fake HTTP server for SDK failure-mode experiments (no external network)."""

from __future__ import annotations

import asyncio
import json
import socket
import struct
from dataclasses import dataclass, field
from typing import Any


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class Stats:
    connections: int = 0
    requests: int = 0
    paths: list[str] = field(default_factory=list)
    bodies: list[bytes] = field(default_factory=list)


async def _read_request(reader: asyncio.StreamReader) -> tuple[str, bytes]:
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    path = lines[0].split(" ")[1] if lines and " " in lines[0] else "?"
    length = 0
    for line in lines[1:]:
        if line.lower().startswith("content-length:"):
            length = int(line.split(":", 1)[1].strip())
    body = await reader.readexactly(length) if length else b""
    return path, body


def _chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


def sse(event: str | None, data: Any) -> bytes:
    payload = data if isinstance(data, str) else json.dumps(data)
    out = ""
    if event:
        out += f"event: {event}\n"
    out += f"data: {payload}\n\n"
    return out.encode()


async def start(behavior: str, **opts: Any) -> tuple[asyncio.AbstractServer, int, Stats]:
    """behavior: hang | disconnect | reset | status | sse"""
    stats = Stats()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        stats.connections += 1
        try:
            if behavior == "reset_on_accept":
                sock = writer.get_extra_info("socket")
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                writer.transport.abort()
                return
            path, body = await _read_request(reader)
            stats.requests += 1
            stats.paths.append(path)
            stats.bodies.append(body)
            if behavior == "hang":
                await asyncio.sleep(opts.get("seconds", 30))
            elif behavior == "disconnect":
                writer.close()
            elif behavior == "reset":
                sock = writer.get_extra_info("socket")
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                writer.transport.abort()
            elif behavior == "status":
                payload = json.dumps(opts["body"]).encode()
                headers = {
                    "content-type": "application/json",
                    "content-length": str(len(payload)),
                    **opts.get("headers", {}),
                }
                head = f"HTTP/1.1 {opts['status']} X\r\n" + "".join(
                    f"{k}: {v}\r\n" for k, v in headers.items()
                )
                writer.write(head.encode() + b"\r\n" + payload)
                await writer.drain()
                writer.close()
            elif behavior == "sse":
                head = (
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
                    "transfer-encoding: chunked\r\n\r\n"
                )
                writer.write(head.encode())
                await writer.drain()
                for item in opts["events"]:
                    writer.write(_chunk(item))
                    await writer.drain()
                    await asyncio.sleep(0.02)
                end = opts.get("end", "finish")
                if end == "finish":
                    writer.write(b"0\r\n\r\n")
                    await writer.drain()
                    writer.close()
                elif end == "drop":
                    writer.close()  # FIN without terminating chunk
                elif end == "reset":
                    sock = writer.get_extra_info("socket")
                    sock.setsockopt(
                        socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                    )
                    writer.transport.abort()
                elif end == "hang":
                    await asyncio.sleep(opts.get("seconds", 30))
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except asyncio.CancelledError:
            raise

    port = free_port()
    server = await asyncio.start_server(handle, "127.0.0.1", port)
    return server, port, stats


def describe(exc: BaseException | None, depth: int = 0) -> None:
    """Print the exception chain with explicit cause/context labels and key MRO entries."""
    seen: set[int] = set()
    label = "raised "
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        cls = type(exc)
        bases = [
            f"{b.__module__.split('.')[0]}.{b.__qualname__}"
            for b in cls.__mro__[1:]
            if b not in (object, BaseException, Exception)
        ]
        extra = ""
        if hasattr(exc, "status_code"):
            extra += f" status_code={getattr(exc, 'status_code', None)!r}"
        if hasattr(exc, "retry_after"):
            extra += f" retry_after={getattr(exc, 'retry_after', None)!r}"
        print(f"  {label}: {cls.__module__}.{cls.__qualname__}{extra}")
        print(f"           msg={str(exc)[:110]!r}")
        print(f"           mro={bases}")
        if exc.__cause__ is not None:
            exc, label = exc.__cause__, "cause  "
        elif exc.__context__ is not None and not exc.__suppress_context__:
            exc, label = exc.__context__, "context"
        else:
            exc = None
