"""05 — Tool Calling (Manual).

Define a tool as a plain dict, detect when the model calls it, execute
locally, and send the tool_result back for a final answer.
"""

from ai_arch_toolkit import LLM, tool_result, user

llm = LLM("claude-haiku-4-5-20251001")

weather_tool = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
}

# Simulated weather data
WEATHER_DATA = {
    "london": "14°C, cloudy with light rain",
    "tokyo": "26°C, sunny and humid",
    "new york": "18°C, partly cloudy",
}

messages = [user("What's the weather in Tokyo?")]
response = llm.complete_sync(messages, tools=[weather_tool])

if response.tool_calls:
    tc = response.tool_calls[0]
    city = str(tc.input.get("city", "")).lower()
    weather = WEATHER_DATA.get(city, "Unknown city")
    print(f"[Tool called: {tc.name}(city={city!r}) → {weather}]")

    # Send the tool result back
    messages.append(response.to_message())
    messages.append(tool_result(weather, tool_use_id=tc.id, name=tc.name))

    final = llm.complete_sync(messages, tools=[weather_tool])
    print("\nAssistant:", final.text)
else:
    print("Assistant:", response.text)
