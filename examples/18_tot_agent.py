"""18 — Tree of Thoughts (ToT) Agent.

The ToTAgent explores multiple reasoning paths using a search tree.
At each node it generates candidate thoughts, scores them with an
evaluator, and expands the best into the frontier. Supports DFS and BFS.

Best for problems where different reasoning paths may lead to different
solutions (puzzles, creative tasks, multi-step logic).
"""

from ai_arch_toolkit.core import LLM, ToolGroup
from ai_arch_toolkit.toolkit.agents import AgentConfig, ToTAgent, ToTConfig
from ai_arch_toolkit.toolkit.tools import math_eval

tools = ToolGroup(math_eval)
llm = LLM("gpt-4.1-nano")

agent = ToTAgent(
    llm,
    tools,
    config=AgentConfig(
        system="You are a problem solver. Explore different approaches.",
        max_iterations=10,
    ),
    tot=ToTConfig(
        n_candidates=3,
        max_depth=3,
        strategy="dfs",
    ),
)

result = agent.run_sync(
    "A farmer has 17 sheep. All but 9 run away. How many are left? Think through this carefully."
)

print("Answer:", result.answer)
print(f"Stop reason: {result.stop_reason}")
print(f"Cost: ${result.total_cost:.4f}")
