"""19 — LATS Flow (Language Agent Tree Search).

LATS combines Monte Carlo Tree Search (MCTS) with ReAct rollouts.
Each rollout: Select (UCT) → Expand (inner ReAct) → Evaluate → Backpropagate.
Low-scoring rollouts trigger reflection to guide future attempts.

Best for tasks where trial-and-error with learning is valuable.
"""

from ai_arch_toolkit import LLM, State, ToolGroup
from ai_arch_toolkit.toolkit.agents import lats_flow, lats_initial_state
from ai_arch_toolkit.toolkit.tools import math_eval, wikipedia_search

tools = ToolGroup(wikipedia_search, math_eval)
llm = LLM("gpt-4.1-nano")

flow = lats_flow(
    llm,
    tools,
    system="You are a research assistant. Use tools to find and verify answers.",
    n_candidates=3,
    max_rollouts=5,
    exploration_weight=1.41,
)

state = State(operational=lats_initial_state("What is the population of France divided by 1000?"))
result = flow.run_sync(state)

print("Answer:", state["response"].text)
print(f"Steps: {len(result.trace.steps)}")
print(f"Cost: ${result.total_cost:.4f}")
