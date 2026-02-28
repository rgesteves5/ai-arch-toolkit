"""10 — ReAct Agent Streaming.

Use run(stream=True) or run_sync(stream=True) to observe each event as
the agent reasons. This is useful for building live UIs, logging, or
debugging multi-step reasoning.

Uses real toolkit tools: wikipedia_search, wikipedia_article, and math_eval.
"""

from ai_arch_toolkit.core import LLM, ToolGroup
from ai_arch_toolkit.toolkit.agents import AgentConfig, ReActAgent
from ai_arch_toolkit.toolkit.tools import math_eval, wikipedia_article, wikipedia_search

tools = ToolGroup(wikipedia_search, wikipedia_article, math_eval)
llm = LLM("gpt-4.1-nano")

agent = ReActAgent(
    llm,
    tools,
    config=AgentConfig(
        system="Answer the question using the available tools.",
        max_iterations=5,
    ),
)

print("=== Streaming agent events ===\n")

for event in agent.run_sync("Who created Python and how old is the language?", stream=True):
    if event.type == "step_start":
        print(f"--- Step {event.step} ---")
    elif event.type == "tool_call":
        print(f"  Tool call: {event.tool_name}({event.tool_args})")
    elif event.type == "tool_result":
        print(f"  Result: {event.result}")
    elif event.type == "error":
        print(f"  Error: {event.error}")
    elif event.type == "step_end":
        if event.response:
            print(f"  Response: {event.response.text}")
        if event.stop_reason:
            print(f"  [stop_reason={event.stop_reason}]")

print("\nDone.")
