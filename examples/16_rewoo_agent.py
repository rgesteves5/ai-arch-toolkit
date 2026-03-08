"""16 — ReWOO Flow (Reasoning WithOut Observation).

ReWOO separates planning from execution. The flow first produces a full
plan with #E{n} placeholders for tool calls, then executes all tools in
order, and finally solves the task using the collected evidence.

This avoids interleaving reasoning with observations — fewer LLM calls.
"""

from ai_arch_toolkit import LLM, State, ToolGroup
from ai_arch_toolkit.toolkit.agents import rewoo_flow, rewoo_initial_state
from ai_arch_toolkit.toolkit.tools import geocode, get_weather

tools = ToolGroup(get_weather, geocode)
llm = LLM("gpt-4.1-nano")

flow = rewoo_flow(
    llm,
    tools,
    system="You are a travel assistant. Plan all tool calls upfront.",
)

state = State(
    operational=rewoo_initial_state("What's the weather in Tokyo and what are its coordinates?")
)
result = flow.run_sync(state)

print("Answer:", state["response"].text)
print(f"Steps: {len(result.trace.steps)}")
print(f"Cost: ${result.total_cost:.4f}")
