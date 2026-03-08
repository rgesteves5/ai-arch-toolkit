"""17 — Plan-Execute Flow.

The plan_execute_flow works in three phases:
1. Plan — generates a numbered step list
2. Execute — runs each step via an inner ReAct loop
3. Solve — synthesizes all step results into a final answer

Supports replanning on failure (max_replans > 0).
"""

from ai_arch_toolkit import LLM, State, ToolGroup
from ai_arch_toolkit.toolkit.agents import plan_execute_flow, plan_execute_initial_state
from ai_arch_toolkit.toolkit.tools import geocode, get_weather, math_eval

tools = ToolGroup(get_weather, geocode, math_eval)
llm = LLM("gpt-4.1-nano")

flow = plan_execute_flow(
    llm,
    tools,
    system="You are a helpful assistant. Break complex tasks into steps.",
    max_replans=1,
)

state = State(
    operational=plan_execute_initial_state(
        "Get the coordinates of Berlin and Tokyo, then calculate the sum of their latitudes."
    )
)
result = flow.run_sync(state)

print("Answer:", state["response"].text)
print(f"Steps: {len(result.trace.steps)}")
print(f"Cost: ${result.total_cost:.4f}")
