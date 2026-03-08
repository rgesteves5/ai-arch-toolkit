"""15 — Reflexion Flow.

The reflexion_flow wraps a react_flow in a retry loop with self-critique.
After each attempt, an evaluator scores the answer. If below threshold,
the flow reflects on what went wrong and retries with that insight.
"""

from ai_arch_toolkit import LLM, State, ToolGroup
from ai_arch_toolkit.toolkit.agents import reflexion_flow, reflexion_initial_state
from ai_arch_toolkit.toolkit.tools import math_eval, wikipedia_search

tools = ToolGroup(wikipedia_search, math_eval)
llm = LLM("gpt-4.1-nano")


def evaluator(task: str, answer: str) -> float:
    """Simple evaluator — checks if the answer looks complete."""
    if not answer or len(answer) < 10:
        return 0.2
    return 0.8


flow = reflexion_flow(
    llm,
    tools,
    evaluator=evaluator,
    threshold=0.7,
    max_retries=2,
    system="You are a research assistant. Use tools to find accurate answers.",
    max_iterations=5,
)

state = State(
    operational=reflexion_initial_state(
        "What year was Python created and what is 2024 minus that year?"
    )
)
result = flow.run_sync(state)

print("Answer:", state["response"].text)
print(f"Steps: {len(result.trace.steps)}")
print(f"Cost: ${result.total_cost:.4f}")
