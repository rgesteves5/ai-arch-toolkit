"""31 — LLMCompiler Agent (OpenAI).

LLMCompilerAgent plans a DAG of tool calls and executes ready nodes in parallel.
"""

from ai_arch_toolkit import AgentConfig, Client, LLMCompilerAgent, ToolRegistry, tool

registry = ToolRegistry()


@tool(registry=registry)
def lookup_planet(input: str) -> str:
    """Return canned planet diameter facts."""
    facts = {
        "earth diameter": "Earth diameter is 12,742 km.",
        "venus diameter": "Venus diameter is 12,104 km.",
        "mars diameter": "Mars diameter is 6,779 km.",
    }
    for key, value in facts.items():
        if key in input.lower():
            return value
    return f"No data for: {input}"


@tool(registry=registry)
def calculator(input: str) -> str:
    """Evaluate a basic arithmetic expression."""
    try:
        return str(eval(input, {"__builtins__": {}}))
    except Exception as exc:
        return f"Error: {exc}"


def on_event(event) -> None:
    if event.type == "plan_created":
        print("\n[plan_created]")
        print(event.result)
    elif event.type == "tool_call":
        print(f"[tool_call] {event.tool_name} args={event.tool_args}")
    elif event.type == "tool_result":
        print(f"[tool_result] {event.result}")
    elif event.type == "step_start":
        print(f"\n[step {event.step_number} start]")


client = Client("openai", model="gpt-5-nano")
agent = LLMCompilerAgent(
    client,
    registry,
    config=AgentConfig(max_iterations=6, on_event=on_event, parallel_tool_execution=True),
)

task = (
    "Find Earth and Mars diameters, then calculate the ratio Earth/Mars. "
    "Return a short explanation with the numeric ratio."
)
result = agent.run(task, max_replans=1)

print("\nFinal answer:")
print(result.answer)
print(f"\nSteps recorded: {len(result.steps)}")
print(f"Total tokens: {result.total_usage.total_tokens}")
