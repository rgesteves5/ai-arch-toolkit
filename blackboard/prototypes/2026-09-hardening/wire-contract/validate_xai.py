"""Contract check for the xAI adapter: replay the captured kwargs into the REAL ``chat.create()``.

``create()`` only builds a ``GetCompletionsRequest`` protobuf (no I/O until ``.sample()``), so
protobuf itself is the validator: unknown fields raise ValueError, wrong types raise TypeError.
"""
from __future__ import annotations

import asyncio
import inspect
import statistics
import time
from typing import Any

import xai_sdk
from xai_sdk import chat as xai_chat
from xai_sdk.aio import chat as aio_chat

from capture import capture

_SIG = inspect.signature(aio_chat.Client.create)


def check_xai(client: Any, kwargs: dict[str, Any]) -> tuple[list[str], Any]:
    try:
        _SIG.bind(None, **kwargs)
    except TypeError as exc:
        return [f"signature | {exc}"], None
    try:
        chat = client.chat.create(**kwargs)
        proto = chat.proto
        proto.SerializeToString()  # the bytes gRPC would put on the wire
        return [], proto
    except Exception as exc:
        return [f"{type(exc).__name__} | {str(exc)[:240]}"], None


async def main() -> None:
    t0 = time.perf_counter()
    client = xai_sdk.AsyncClient(api_key="offline")  # lazy gRPC channel: no connection attempt
    print(f"real AsyncClient built offline in {(time.perf_counter() - t0) * 1000:.0f} ms")
    timings = []
    for model in ("grok-4", "grok-4.20-multi-agent"):
        print(f"##### model={model}")
        for cap in await capture("xai", model=model):
            if cap.kwargs is None:
                print(f"  skip  {cap.scenario:24s} adapter raised: {cap.error[:90]}")
                continue
            t0 = time.perf_counter()
            errs, proto = check_xai(client, cap.kwargs)
            timings.append((time.perf_counter() - t0) * 1000)
            fields = [f.name for f, _ in proto.ListFields()] if proto is not None else None
            print(f"  {'FAIL' if errs else 'pass'}  {cap.scenario:24s} proto fields={fields}")
            for e in errs:
                print("           ", e)
    print(f"per-validation: median {statistics.median(timings):.2f} ms, max {max(timings):.2f} ms")

    print("##### what protobuf construction rejects on its own")
    pb = xai_chat.chat_pb2
    for label, fn in {
        "Message(bogus=1)": lambda: pb.Message(bogus=1),
        "Message(role='assistant')  # str for enum": lambda: pb.Message(role="assistant"),
        "Message(role=999)  # unknown enum number": lambda: pb.Message(role=999),
        "ToolCall(id=5)  # int for str": lambda: pb.ToolCall(id=5),
        "FunctionCall(arguments={'a': 1})  # dict for str": lambda: pb.FunctionCall(arguments={"a": 1}),
        "ToolChoice(**{'type': 'function'})": lambda: pb.ToolChoice(**{"type": "function"}),
        "GetCompletionsRequest(temperature='hot')": lambda: pb.GetCompletionsRequest(temperature="hot"),
        "GetCompletionsRequest(max_tokens=-1)  # range": lambda: pb.GetCompletionsRequest(max_tokens=-1),
    }.items():
        try:
            fn()
            print(f"  accepted  {label}")
        except Exception as exc:
            print(f"  REJECTED  {label}: {type(exc).__name__}: {str(exc)[:110]}")
    await client.close()


asyncio.run(main())
