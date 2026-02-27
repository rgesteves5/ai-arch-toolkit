"""09 — ReAct Agent.

The ReActAgent automates the tool loop from example 06. Give it an LLM,
a ToolGroup, and a task — it handles the Thought → Action → Observation
cycle until the model produces a final answer or a stop condition fires.
"""

from ai_arch_toolkit.agents import AgentConfig, ReActAgent
from ai_arch_toolkit.core import LLM, ToolGroup, tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: City name.
    """
    data = {
        "london": "14°C, cloudy with light rain",
        "tokyo": "26°C, sunny and humid",
        "new york": "18°C, partly cloudy",
    }
    return data.get(city.lower(), f"No data for {city}")


@tool
def get_population(city: str) -> str:
    """Get the population of a city.

    Args:
        city: City name.
    """
    data = {
        "london": "8.8 million",
        "tokyo": "13.9 million",
        "new york": "8.3 million",
    }
    return data.get(city.lower(), f"No data for {city}")


tools = ToolGroup(get_weather, get_population)
llm = LLM("gpt-4.1-nano")

agent = ReActAgent(
    llm,
    tools,
    config=AgentConfig(
        system="You are a helpful travel assistant. Use the tools to answer.",
        max_iterations=5,
    ),
)

result = agent.run_sync("What's the weather and population in Tokyo?")

print("Answer:", result.answer)
print(f"Steps: {len(result.steps)}")
print(f"Stop reason: {result.stop_reason}")
print(f"Tokens — in: {result.total_usage.input_tokens}, out: {result.total_usage.output_tokens}")
print(f"Cost: ${result.total_cost:.4f}")
