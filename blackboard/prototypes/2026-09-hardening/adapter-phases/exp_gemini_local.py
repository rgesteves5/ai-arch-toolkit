"""google-genai: errors raised locally inside generate_content (LOOPBACK client only)."""

from __future__ import annotations

import asyncio
import sys
import warnings

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import fakeserver as fs  # noqa: E402
from google import genai  # noqa: E402

from ai_arch_toolkit.core._providers._gemini import GeminiProvider  # noqa: E402

warnings.simplefilter("ignore")


async def main() -> None:
    srv, port, st = await fs.start("status", status=200, body={"candidates": []})
    g = GeminiProvider("gemini-2.5-flash", "dummy", timeout=1.0)
    g._client = genai.Client(  # loopback ONLY — set before any call
        api_key="dummy",
        http_options={"base_url": f"http://127.0.0.1:{port}", "retry_options": {"attempts": 1}, "timeout": 1000},
    )
    for title, msgs in (
        ("empty message list", []),
        ("only a system message (contents empty after extraction)", [{"role": "system", "content": "s"}]),
    ):
        print(f"\n=== gemini.complete: {title}")
        before = st.requests
        try:
            r = await g.complete(msgs)
            print("  no exception ->", repr(r)[:80])
        except BaseException as exc:  # noqa: BLE001
            fs.describe(exc)
        print(f"  requests sent to the loopback server by this call: {st.requests - before}")
    srv.close()


asyncio.run(main())
