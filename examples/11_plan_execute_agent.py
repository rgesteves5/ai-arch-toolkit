"""11 — Plan-then-Execute Agent (Gemini).

The PlanExecuteAgent first creates a multi-step plan, then executes
each step using available tools.
"""

from ai_arch_toolkit import (
    AgentConfig,
    Client,
    PlanExecuteAgent,
    ToolRegistry,
    tool,
)

registry = ToolRegistry()


@tool(registry=registry)
def lookup(topic: str) -> str:
    """Look up factual information about a topic.

    Args:
        topic: The topic to look up.
    """
    facts = {
        "mars": "Mars is the 4th planet, with a diameter of 6,779 km.",
        "earth": "Earth is the 3rd planet, with a diameter of 12,742 km.",
        "jupiter": "Jupiter is the 5th planet, with a diameter of 139,820 km.",
    }
    for key, value in facts.items():
        if key in topic.lower():
            return value
    return f"No data for: {topic}"


@tool(registry=registry)
def compare(item_a: str, item_b: str) -> str:
    """Compare two items and return a brief comparison.

    Args:
        item_a: First item to compare.
        item_b: Second item to compare.
    """
    return f"Comparing {item_a} vs {item_b}: see individual lookups for details."


def on_event(event):
    if event.type == "plan_created":
        print("[Plan created]")
    elif event.type == "step_start":
        print(f"  Executing step {event.step_number}...")
    elif event.type == "tool_call":
        print(f"    Tool: {event.tool_name}({event.tool_args})")
    elif event.type == "tool_result":
        print(f"    Result: {event.result[:80]}")


client = Client("gemini", model="gemini-2.0-flash")
agent = PlanExecuteAgent(
    client,
    registry,
    config=AgentConfig(max_iterations=8, on_event=on_event),
)

result = agent.run("Compare the sizes of Mars and Jupiter.")

print(f"\nAnswer: {result.answer}")
print(f"Steps: {len(result.steps)}")
