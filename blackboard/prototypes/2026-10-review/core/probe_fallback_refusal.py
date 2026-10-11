"""A fallback that refuses the request at preparation ends the chain (no network)."""

from __future__ import annotations

import pathlib
import asyncio
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))  # the repo root, 4 levels up

from ai_arch_toolkit.core import RequestError
from ai_arch_toolkit.core._exceptions import APIError
from tests.fake_provider import fake_llm


async def main() -> None:
    c, pc = fake_llm(model="claude-sonnet-4-6")
    b, pb = fake_llm(model="claude-sonnet-4-6")
    pb.refusals = [RequestError("model-b takes thinking_effort in [...], not 'max'")]
    a, pa = fake_llm(APIError(503, "overloaded"), model="claude-sonnet-4-6", fallback=[b, c])
    try:
        r = await a.complete("hi")
        print("answered:", r.text, "| calls a/b/c:", pa.calls, pb.calls, pc.calls)
    except Exception as exc:  # noqa: BLE001
        print("raised:", type(exc).__name__, exc, "| calls a/b/c:", pa.calls, pb.calls, pc.calls)


asyncio.run(main())
