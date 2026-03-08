"""18 — Tree of Thoughts (ToT) Flow.

The tot_flow explores multiple reasoning paths using a search tree.
At each node it generates candidate thoughts, scores them with an
evaluator, and expands the best into the frontier. Supports DFS and BFS.

Best for problems where different reasoning paths may lead to different
solutions (puzzles, creative tasks, multi-step logic).
"""

from ai_arch_toolkit import LLM, State, ToolGroup
from ai_arch_toolkit.toolkit.agents import tot_flow, tot_initial_state
from ai_arch_toolkit.toolkit.tools import math_eval

tools = ToolGroup(math_eval)
llm = LLM("gpt-4.1-nano")

flow = tot_flow(
    llm,
    tools,
    system="You are a problem solver. Explore different approaches.",
    n_candidates=3,
    max_depth=3,
    max_iterations=10,
    strategy="dfs",
)

state = State(
    operational=tot_initial_state(
        "A farmer has 17 sheep. All but 9 run away. "
        "How many are left? Think through this carefully."
    )
)
result = flow.run_sync(state)

print("Answer:", state["response"].text)
print(f"Steps: {len(result.trace.steps)}")
print(f"Cost: ${result.total_cost:.4f}")
