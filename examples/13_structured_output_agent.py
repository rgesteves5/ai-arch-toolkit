"""13 — Structured Output Agent.

A ReActAgent that uses tools to gather data and returns a typed JSON
weather report via OutputSchema.
"""

from ai_arch_toolkit.core import LLM, OutputSchema, ToolGroup
from ai_arch_toolkit.toolkit.agents import AgentConfig, ReActAgent
from ai_arch_toolkit.toolkit.tools import geocode, get_weather

# Define the output schema for a weather report
weather_schema = OutputSchema(
    name="WeatherReport",
    schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "latitude": {"type": "number"},
            "longitude": {"type": "number"},
            "temperature_c": {"type": "number", "description": "Temperature in Celsius"},
            "summary": {"type": "string", "description": "Brief weather summary"},
        },
        "required": ["city", "latitude", "longitude", "temperature_c", "summary"],
        "additionalProperties": False,
    },
    strict=True,
)

tools = ToolGroup(get_weather, geocode)
llm = LLM("gpt-4.1-nano")

agent = ReActAgent(
    llm,
    tools,
    config=AgentConfig(
        system="You are a weather assistant. Use the tools to gather data, then produce a report.",
        max_iterations=5,
        output_schema=weather_schema,
    ),
)

result = agent.run_sync("Give me a weather report for Berlin.")

print("Answer:", result.answer)
if result.parsed:
    print(f"City: {result.parsed['city']}")
    print(f"Temp: {result.parsed['temperature_c']}°C")
    print(f"Summary: {result.parsed['summary']}")
print(f"Steps: {len(result.steps)}")
print(f"Cost: ${result.total_cost:.4f}")
