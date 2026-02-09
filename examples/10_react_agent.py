"""10 — ReAct Agent (OpenAI).

A ReActAgent that reasons through Thought → Action → Observation loops,
with event callbacks for full observability.
"""

from ai_arch_toolkit import (
    AgentConfig,
    Client,
    ReActAgent,
    ToolRegistry,
    tool,
)

registry = ToolRegistry()


@tool(registry=registry)
def search(query: str) -> str:
    """Search for information on a topic.

    Args:
        query: The search query.
    """
    data = {
        "python creator": "Python was created by Guido van Rossum, first released in 1991.",
        "eiffel tower height": "The Eiffel Tower is 330 metres (1,083 ft) tall.",
        "speed of light": "The speed of light is approximately 299,792,458 m/s.",
    }
    for key, value in data.items():
        if key in query.lower():
            return value
    return f"No results found for: {query}"


@tool(registry=registry)
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression, e.g. "330 * 3.281".
    """
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


def on_event(event):
    """Print agent events for observability."""
    if event.type == "step_start":
        print(f"\n--- Step {event.step_number} ---")
    elif event.type == "tool_call":
        print(f"  Action: {event.tool_name}({event.tool_args})")
    elif event.type == "tool_result":
        print(f"  Observation: {event.result[:100]}")


client = Client("openai", model="gpt-5-nano")
agent = ReActAgent(
    client,
    registry,
    config=AgentConfig(max_iterations=5, on_event=on_event),
)

result = agent.run("How tall is the Eiffel Tower in feet? Use the search tool first.")

print(f"\nFinal answer: {result.answer}")
print(f"Steps taken: {len(result.steps)}")
print(f"Total tokens: {result.total_usage.total_tokens}")
