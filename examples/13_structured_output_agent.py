"""13 — Structured Output with ReAct Flow.

A react_flow that uses tools to gather data and returns a typed JSON
weather report via OutputSchema passed as an llm_kwarg.
"""

from ai_arch_toolkit import LLM, OutputSchema, State, ToolGroup
from ai_arch_toolkit.toolkit.agents import react_flow, react_initial_state
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

flow = react_flow(
    llm,
    tools,
    system="You are a weather assistant. Use the tools to gather data, then produce a report.",
    max_iterations=5,
    llm_kwargs={"output_schema": weather_schema},
)

state = State(operational=react_initial_state("Give me a weather report for Berlin."))
result = flow.run_sync(state)

response = state["response"]
print("Answer:", response.text)
if response.parsed:
    print(f"City: {response.parsed['city']}")
    print(f"Temp: {response.parsed['temperature_c']}°C")
    print(f"Summary: {response.parsed['summary']}")
print(f"Steps: {len(result.trace.steps)}")
print(f"Cost: ${result.total_cost:.4f}")
