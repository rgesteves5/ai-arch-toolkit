"""28 — LATS Agent (OpenAI).

Demonstrates LATSAgent (Language Agent Tree Search) with local tools.
"""

from ai_arch_toolkit import AgentConfig, Client, LATSAgent, ToolRegistry, tool

registry = ToolRegistry()


@tool(registry=registry)
def search(query: str) -> str:
    """Return canned facts for known queries."""
    data = {
        "eiffel tower height meters": "The Eiffel Tower height is 330 meters.",
        "meters to feet factor": "Use factor 3.28084 feet per meter.",
    }
    for key, value in data.items():
        if key in query.lower():
            return value
    return f"No result for: {query}"


@tool(registry=registry)
def calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as exc:
        return f"Error: {exc}"


def on_event(event) -> None:
    if event.type == "step_start":
        print(f"\n[MCTS iteration {event.step_number}]")
    elif event.type == "reflection":
        print(f"[reflection] {event.result[:120]}")


def evaluator(answer: str) -> float:
    text = answer.lower()
    if "1082" in text or "1083" in text:
        return 0.95
    if answer and not answer.startswith("["):
        return 0.6
    return 0.0


client = Client("openai", model="gpt-5-nano")
agent = LATSAgent(
    client,
    registry,
    config=AgentConfig(max_iterations=3, on_event=on_event),
)

task = "How tall is the Eiffel Tower in feet? Use tools to compute it."
result = agent.run(task, num_expansions=2, evaluator=evaluator)

print("\nFinal answer:")
print(result.answer)
print(f"\nSteps recorded: {len(result.steps)}")
print(f"Total tokens: {result.total_usage.total_tokens}")
