"""09 — ReAct Flow.

The react_flow automates the tool loop from example 06. Give it an LLM,
a ToolGroup, and a task — it handles the Thought → Action → Observation
cycle until the model produces a final answer or a stop condition fires.

Uses real toolkit tools: get_weather (Open-Meteo) and geocode (free API).
"""

from ai_arch_toolkit import LLM, State, ToolGroup
from ai_arch_toolkit.toolkit.agents import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.tools import geocode, get_weather

tools = ToolGroup(get_weather, geocode)
llm = LLM("gpt-4.1-nano")

flow = react_flow(
    llm,
    tools,
    system="You are a helpful travel assistant. Use the tools to answer.",
    max_iterations=5,
)

state = State(operational=react_initial_state("What's the weather and coordinates of Tokyo?"))
result = flow.run_sync(state)

print("Answer:", state["response"].text)
print(f"Steps: {len(result.trace.steps)}")
print(f"Duration: {result.total_duration:.2f}s")
print(f"Cost: ${result.total_cost:.4f}")
