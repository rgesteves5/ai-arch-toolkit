"""TracingMiddleware rebuilds the Request without its ``image``: generate_image breaks (no network)."""

from __future__ import annotations

import pathlib
import asyncio
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))  # the repo root, 4 levels up

from ai_arch_toolkit.core import TracingMiddleware
from tests.fake_provider import fake_llm


async def main() -> None:
    for middleware in ([], [TracingMiddleware()]):
        llm, provider = fake_llm(images=True, middleware=middleware)
        try:
            await llm.generate_image("a cat")
            last = provider.last
            print(f"{len(middleware)} mw: sent image={last.image is not None} kwargs={sorted(last.kwargs)}")
        except Exception as exc:  # noqa: BLE001
            print(f"{len(middleware)} mw: raised {type(exc).__name__}: {exc}")


asyncio.run(main())
