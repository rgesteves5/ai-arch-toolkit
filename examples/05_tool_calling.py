"""05 — Tool Calling (Anthropic).

Manually define a Tool, detect when the model calls it, execute locally,
and send the ToolResult back for a final answer.
"""

from ai_arch_toolkit import Client, Message, Tool, ToolResult

client = Client("anthropic", model="claude-haiku-4-5")

weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
)

# Simulated weather data
WEATHER_DATA = {
    "london": "14°C, cloudy with light rain",
    "tokyo": "26°C, sunny and humid",
    "new york": "18°C, partly cloudy",
}

messages = [Message(role="user", content="What's the weather in Tokyo?")]
response = client.chat(messages, tools=[weather_tool])

if response.tool_calls:
    tc = response.tool_calls[0]
    city = tc.arguments["city"].lower()
    weather = WEATHER_DATA.get(city, "Unknown city")
    print(f"[Tool called: {tc.name}(city={city!r}) → {weather}]")

    # Send the tool result back
    messages.append(response.to_message())
    messages.append(ToolResult(tool_call_id=tc.id, name=tc.name, content=weather))

    final = client.chat(messages, tools=[weather_tool])
    print("\nAssistant:", final.text)
else:
    print("Assistant:", response.text)
