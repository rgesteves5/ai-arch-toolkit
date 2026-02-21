"""21 — Stream Events Deep Dive (OpenAI).

Demonstrates robust handling of ``stream_events()``:
  - text
  - tool_call
  - thinking
  - usage
  - done
"""

from ai_arch_toolkit import Client, Tool

client = Client("openai", model="gpt-5-nano")

tools = [
    Tool(
        name="get_weather",
        description="Return weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    ),
    Tool(
        name="calculator",
        description="Evaluate a short arithmetic expression.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    ),
]

prompt = (
    "Plan a short outdoor itinerary for Paris this weekend. "
    "If useful, call tools for weather and basic math."
)

print("=== stream_events deep dive ===")
event_counts: dict[str, int] = {}
for event in client.stream_events(prompt, tools=tools):
    event_counts[event.type] = event_counts.get(event.type, 0) + 1

    if event.type == "text":
        print(event.text, end="", flush=True)
    elif event.type == "tool_call" and event.tool_call is not None:
        tool_call = event.tool_call
        print(f"\n\n[tool_call] {tool_call.name} args={tool_call.arguments}")
    elif event.type == "thinking" and event.thinking:
        print(f"\n\n[thinking] {event.thinking}")
    elif event.type == "usage" and event.usage is not None:
        usage = event.usage
        print(
            "\n\n[usage] "
            f"input={usage.input_tokens} output={usage.output_tokens} total={usage.total_tokens}"
        )
    elif event.type == "done":
        print("\n\n[done]")
    else:
        print(f"\n\n[event:{event.type}]")

print("\nEvent counts:")
for event_type, count in sorted(event_counts.items()):
    print(f"  {event_type}: {count}")
