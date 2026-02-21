"""29 — Reflexion Agent (OpenAI).

ReflexionAgent retries with self-reflection until an evaluator threshold is met.
"""

from ai_arch_toolkit import AgentConfig, Client, ReflexionAgent, ToolRegistry, tool

registry = ToolRegistry()


@tool(registry=registry)
def search(query: str) -> str:
    """Return simple physics facts."""
    data = {
        "speed of light m/s": "The speed of light is 299,792,458 m/s.",
        "convert m/s to km/s": "To convert m/s to km/s, divide by 1000.",
    }
    for key, value in data.items():
        if key in query.lower():
            return value
    return f"No results for: {query}"


@tool(registry=registry)
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as exc:
        return f"Error: {exc}"


def on_event(event) -> None:
    if event.type == "step_start":
        print(f"\n[attempt {event.step_number}]")
    elif event.type == "reflection":
        print(f"[reflection] {event.result[:140]}")


def evaluator(answer: str) -> float:
    normalized = answer.lower()
    if "299" in normalized and ("km/s" in normalized or "km per second" in normalized):
        return 0.95
    if answer.strip():
        return 0.4
    return 0.0


client = Client("openai", model="gpt-5-nano")
agent = ReflexionAgent(
    client,
    registry,
    config=AgentConfig(max_iterations=3, on_event=on_event),
)

task = "What is the speed of light in km/s? Show one short calculation."
result = agent.run(task, evaluator=evaluator, threshold=0.9)

print("\nFinal answer:")
print(result.answer)
print(f"\nSteps recorded: {len(result.steps)}")
print(f"Total tokens: {result.total_usage.total_tokens}")
