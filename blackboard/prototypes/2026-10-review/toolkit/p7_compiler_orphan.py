import pathlib
import sys, asyncio
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4] / "tests"))  # the repo root, 4 levels up
from fake_provider import fake_llm, Reply
from ai_arch_toolkit.core import State, Response, Usage, ToolGroup
from ai_arch_toolkit.toolkit.agents.flows import llm_compiler_flow, llm_compiler_initial_state
from ai_arch_toolkit.toolkit.budget import BudgetPolicy

U = Usage(input_tokens=10, output_tokens=5)
M = "claude-sonnet-4-6"
plan = Response(text="$1. A [deps: none]\n$2. B [deps: none]", usage=U, model=M)
planner, pp = fake_llm(plan)
executor, ep = fake_llm(Reply(response=Response(text="A done", usage=U, model=M), delay=0.5),
                        Response(text="B done", usage=U, model=M))

async def main():
    flow = llm_compiler_flow(planner, ToolGroup(), exec_llm=executor)
    state = State(operational=llm_compiler_initial_state("task"))
    res = await flow.run(state, budget_policy=BudgetPolicy(max_llm_calls=2))
    print("run returned; trace:", [(s.name, (s.error or "")[:60]) for s in res.trace.steps])
    print("executor calls in flight at return:", ep.live, "| meter llm_calls:", res.meter.llm_calls)
    await asyncio.sleep(1.0)
    print("after 1s: executor live:", ep.live, "| executor calls:", ep.calls, "| meter llm_calls:", res.meter.llm_calls)
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    print("leftover tasks:", len(tasks))

asyncio.run(main())
