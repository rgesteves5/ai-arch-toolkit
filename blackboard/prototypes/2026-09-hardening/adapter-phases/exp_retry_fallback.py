"""Under a cost cap: does a retry / fallback survive the first failed attempt? (loopback, dummy keys)"""

from __future__ import annotations

import asyncio
import sys
import warnings

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import fakeserver as fs  # noqa: E402

from ai_arch_toolkit.core import LLM, MeterScope, RetryConfig, RunConfig  # noqa: E402
from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetExceeded, BudgetPolicy  # noqa: E402
from ai_arch_toolkit.core._metering._admission import AdmissionDenied  # noqa: E402

warnings.simplefilter("ignore")
print("BudgetExceeded MRO:", [c.__name__ for c in BudgetExceeded.__mro__][:4], "| AdmissionDenied base:", AdmissionDenied.__mro__[1].__name__)

BODY = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}


async def main() -> None:
    for capped in (False, True):
        srv, port, st = await fs.start("status", status=429, body=BODY, headers={"retry-after": "0.01"})
        srv2, port2, st2 = await fs.start("status", status=429, body=BODY, headers={"retry-after": "0.01"})
        cfg = RunConfig(controller=BudgetController(BudgetPolicy(max_cost=1.0))) if capped else RunConfig()
        with MeterScope(cfg) as scope:
            fb = LLM("claude-haiku-4-5", api_key="dummy", base_url=f"http://127.0.0.1:{port2}")
            llm = LLM(
                "claude-sonnet-4-5", api_key="dummy", base_url=f"http://127.0.0.1:{port}",
                retry=RetryConfig(max_retries=2, base_delay=0.01), fallback=fb,
            )
            try:
                await llm.complete("hi")
            except Exception as exc:  # noqa: BLE001
                print(f"\ncost cap={capped}: raised {type(exc).__name__}: {str(exc)[:80]}")
            s = scope.snapshot()
            print(f"  primary server requests={st.requests} fallback server requests={st2.requests} "
                  f"llm_calls={s.llm_calls} unknown={s.unknown_cost_count}")
            await llm.close()
            await fb.close()
        srv.close()
        srv2.close()


asyncio.run(main())
