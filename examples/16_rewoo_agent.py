"""16 — ReWOO Agent (Reasoning WithOut Observation).

ReWOO separates planning from execution. The agent first produces a full
plan with #E{n} placeholders for tool calls, then executes all tools in
order, and finally solves the task using the collected evidence.

This avoids interleaving reasoning with observations — fewer LLM calls.
"""

from ai_arch_toolkit.core import LLM, ToolGroup
from ai_arch_toolkit.toolkit.agents import AgentConfig, ReWOOAgent
from ai_arch_toolkit.toolkit.tools import geocode, get_weather

tools = ToolGroup(get_weather, geocode)
llm = LLM("gpt-4.1-nano")

agent = ReWOOAgent(
    llm,
    tools,
    config=AgentConfig(
        system="You are a travel assistant. Plan all tool calls upfront.",
        max_iterations=5,
    ),
)

result = agent.run_sync("What's the weather in Tokyo and what are its coordinates?")

print("Answer:", result.answer)
print(f"Steps: {len(result.steps)}")
print(f"Stop reason: {result.stop_reason}")
print(f"Cost: ${result.total_cost:.4f}")
