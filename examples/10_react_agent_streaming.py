"""10 — ReAct Agent Streaming.

Use run(stream=True) or run_sync(stream=True) to observe each event as
the agent reasons. This is useful for building live UIs, logging, or
debugging multi-step reasoning.
"""

from ai_arch_toolkit.agents import AgentConfig, ReActAgent
from ai_arch_toolkit.core import LLM, ToolGroup, tool


@tool
def search(query: str) -> str:
    """Search for information.

    Args:
        query: The search query.
    """
    results = {
        "python creator": "Guido van Rossum created Python in 1991.",
        "rust creator": "Graydon Hoare created Rust at Mozilla in 2010.",
    }
    for key, value in results.items():
        if key in query.lower():
            return value
    return f"No results for: {query}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression.

    Args:
        expression: A math expression, e.g. "2024 - 1991".
    """
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


tools = ToolGroup(search, calculate)
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
            print(f"  Response: {event.response.text[:100]}...")
        if event.stop_reason:
            print(f"  [stop_reason={event.stop_reason}]")

print("\nDone.")
