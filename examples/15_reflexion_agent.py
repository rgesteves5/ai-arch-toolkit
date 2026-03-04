"""15 — Reflexion Agent.

The ReflexionAgent wraps ReActAgent in a retry loop with self-critique.
After each attempt, an evaluator scores the answer. If below threshold,
the agent reflects on what went wrong and retries with that insight.
"""

from ai_arch_toolkit.core import LLM, ToolGroup
from ai_arch_toolkit.toolkit.agents import AgentConfig, ReflexionAgent, ReflexionConfig
from ai_arch_toolkit.toolkit.tools import math_eval, wikipedia_search

tools = ToolGroup(wikipedia_search, math_eval)
llm = LLM("gpt-4.1-nano")

agent = ReflexionAgent(
    llm,
    tools,
    config=AgentConfig(
        system="You are a research assistant. Use tools to find accurate answers.",
        max_iterations=5,
    ),
    reflexion=ReflexionConfig(
        max_retries=2,
        threshold=0.7,
    ),
)

result = agent.run_sync("What year was Python created and what is 2024 minus that year?")

print("Answer:", result.answer)
print(f"Steps: {len(result.steps)}")
print(f"Stop reason: {result.stop_reason}")
print(f"Cost: ${result.total_cost:.4f}")
