"""10 — ReAct Flow Streaming.

Use flow.iter_sync() to observe each event as the flow executes.
This is useful for building live UIs, logging, or debugging multi-step
reasoning.

Uses real toolkit tools: wiki_search, wiki_read, and math_eval.
"""

from ai_arch_toolkit import LLM, State, ToolGroup
from ai_arch_toolkit.toolkit.agents import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.tools import math_eval, wiki_read, wiki_search

tools = ToolGroup(wiki_search, wiki_read, math_eval)
llm = LLM("gpt-4.1-mini")

flow = react_flow(
    llm,
    tools,
    system="Answer the question using the available tools.",
    max_iterations=5,
)

state = State(operational=react_initial_state("Who created Python and how old is the language?"))

print("=== Streaming flow events ===\n")

for event in flow.iter_sync(state):
    if event.type == "step_start":
        print(f"--- Step: {event.step_name} ---")
    elif event.type == "step_end":
        if event.result and event.result.is_ok:
            print(f"  Completed: {event.step_name}")
    elif event.type == "flow_end":
        print(f"\n[flow complete — cost: ${event.trace.total_cost:.4f}]")

print(f"\nAnswer: {state['response'].text}")
print("\nDone.")
