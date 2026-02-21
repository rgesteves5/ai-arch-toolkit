"""30 — ReWOO Agent (OpenAI).

ReWOOAgent splits reasoning into planner -> worker -> solver phases.
"""

from ai_arch_toolkit import AgentConfig, Client, ReWOOAgent, ToolRegistry, tool

registry = ToolRegistry()


@tool(registry=registry)
def lookup(input: str) -> str:
    """Return canned values for known astronomy facts."""
    facts = {
        "mars diameter": "Mars diameter is 6,779 km.",
        "jupiter diameter": "Jupiter diameter is 139,820 km.",
    }
    for key, value in facts.items():
        if key in input.lower():
            return value
    return f"No lookup data for: {input}"


@tool(registry=registry)
def calculator(input: str) -> str:
    """Evaluate a basic arithmetic expression string."""
    try:
        return str(eval(input, {"__builtins__": {}}))
    except Exception as exc:
        return f"Error: {exc}"


def on_event(event) -> None:
    if event.type == "plan_created":
        print("\n[planner output]")
        print(event.result)
    elif event.type == "tool_call":
        print(f"[tool_call] {event.tool_name} args={event.tool_args}")
    elif event.type == "tool_result":
        print(f"[tool_result] {event.result}")


client = Client("openai", model="gpt-5-nano")
agent = ReWOOAgent(
    client,
    registry,
    config=AgentConfig(max_iterations=5, on_event=on_event),
)

task = (
    "Using the tools, find Mars and Jupiter diameters, then compute how many times "
    "larger Jupiter is than Mars."
)
result = agent.run(task)

print("\nFinal answer:")
print(result.answer)
print(f"\nSteps recorded: {len(result.steps)}")
print(f"Total tokens: {result.total_usage.total_tokens}")
