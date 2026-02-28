"""17 — Plan-Execute Agent.

The PlanExecuteAgent works in three phases:
1. Plan — generates a numbered step list
2. Execute — runs each step via an inner ReActAgent
3. Solve — synthesizes all step results into a final answer

Supports replanning on failure (max_replans > 0).
"""

from ai_arch_toolkit.core import LLM, ToolGroup
from ai_arch_toolkit.toolkit.agents import AgentConfig, PlanExecuteAgent, PlanExecuteConfig
from ai_arch_toolkit.toolkit.tools import geocode, get_weather, math_eval

tools = ToolGroup(get_weather, geocode, math_eval)
llm = LLM("gpt-4.1-nano")

agent = PlanExecuteAgent(
    llm,
    tools,
    config=AgentConfig(
        system="You are a helpful assistant. Break complex tasks into steps.",
        max_iterations=10,
    ),
    plan_execute=PlanExecuteConfig(max_replans=1),
)

result = agent.run_sync(
    "Get the coordinates of Berlin and Tokyo, then calculate the sum of their latitudes."
)

print("Answer:", result.answer)
print(f"Steps: {len(result.steps)}")
print(f"Stop reason: {result.stop_reason}")
print(f"Cost: ${result.total_cost:.4f}")
