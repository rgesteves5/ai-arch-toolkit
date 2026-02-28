"""09 — ReAct Agent.

The ReActAgent automates the tool loop from example 06. Give it an LLM,
a ToolGroup, and a task — it handles the Thought → Action → Observation
cycle until the model produces a final answer or a stop condition fires.

Uses real toolkit tools: get_weather (Open-Meteo) and geocode (free API).
"""

from ai_arch_toolkit.core import LLM, ToolGroup
from ai_arch_toolkit.toolkit.agents import AgentConfig, ReActAgent
from ai_arch_toolkit.toolkit.tools import geocode, get_weather

tools = ToolGroup(get_weather, geocode)
llm = LLM("gpt-4.1-nano")

agent = ReActAgent(
    llm,
    tools,
    config=AgentConfig(
        system="You are a helpful travel assistant. Use the tools to answer.",
        max_iterations=5,
    ),
)

result = agent.run_sync("What's the weather and coordinates of Tokyo?")

print("Answer:", result.answer)
print(f"Steps: {len(result.steps)}")
print(f"Stop reason: {result.stop_reason}")
print(f"Tokens — in: {result.total_usage.input_tokens}, out: {result.total_usage.output_tokens}")
print(f"Cost: ${result.total_cost:.4f}")
