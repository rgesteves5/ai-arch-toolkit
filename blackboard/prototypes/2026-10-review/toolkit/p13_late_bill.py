import time, asyncio
from ai_arch_toolkit.core import tool, ToolGroup, ToolCall, MeterScope, RunConfig
from ai_arch_toolkit.core._tools._billing import bill

@tool(timeout_s=0.1)
def slow_search(q: str) -> str:
    """Search."""
    time.sleep(0.3)          # the HTTP request is still in flight at the deadline
    bill("brave_search", 1)  # the service accepted and billed it
    return "results"

async def main():
    group = ToolGroup(slow_search)
    with MeterScope(RunConfig()) as scope:
        r = await group.async_execute(ToolCall(id="1", name="slow_search", input={"q": "x"}))
        print("result:", r.error.type if r.error else r.value)
        await asyncio.sleep(0.5)  # the thread finishes and bills
        s = scope.snapshot()
        print("tool_calls:", s.tool_calls, "| metered cost:", s.cost.to_float(), "| uncertain:", s.uncertain_cost.to_float())
asyncio.run(main())
