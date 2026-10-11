import asyncio
from ai_arch_toolkit.core import State, Step, Result
from ai_arch_toolkit.toolkit.flow import Flow, FlowStep

def make(fail_delay, ok_delay):
    async def fetch(s):
        await asyncio.sleep(fail_delay); return Result(error="fetch failed")
    async def other(s):
        await asyncio.sleep(ok_delay); return Result(value="ok", artifacts={"x": 1})
    async def use(s):
        return Result(value="used")
    inner = Flow(FlowStep(step=Step(name="fetch", fn=fetch)), FlowStep(step=Step(name="other", fn=other)),
                 FlowStep(step=Step(name="use", fn=use), after=("fetch",)), name="inner")
    outer = Flow(inner, Step(name="after_inner", fn=use), name="outer")
    return outer

for fd, od in [(0.0, 0.05), (0.05, 0.0)]:
    res = make(fd, od).run_sync(State())
    print(f"fetch fails at {fd}s, other ok at {od}s -> outer trace:", [(s.name, s.error) for s in res.trace.steps])
