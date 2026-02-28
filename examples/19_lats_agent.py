"""19 — LATS Agent (Language Agent Tree Search).

LATS combines Monte Carlo Tree Search (MCTS) with ReAct rollouts.
Each rollout: Select (UCT) → Expand (inner ReAct) → Evaluate → Backpropagate.
Low-scoring rollouts trigger reflection to guide future attempts.

Best for tasks where trial-and-error with learning is valuable.
"""

from ai_arch_toolkit.core import LLM, ToolGroup
from ai_arch_toolkit.toolkit.agents import AgentConfig, LATSAgent, LATSConfig
from ai_arch_toolkit.toolkit.tools import math_eval, wikipedia_search

tools = ToolGroup(wikipedia_search, math_eval)
llm = LLM("gpt-4.1-nano")

agent = LATSAgent(
    llm,
    tools,
    config=AgentConfig(
        system="You are a research assistant. Use tools to find and verify answers.",
        max_iterations=10,
    ),
    lats=LATSConfig(
        n_candidates=3,
        max_rollouts=5,
        exploration_weight=1.41,
    ),
)

result = agent.run_sync("What is the population of France divided by 1000?")

print("Answer:", result.answer)
print(f"Stop reason: {result.stop_reason}")
print(f"Cost: ${result.total_cost:.4f}")
