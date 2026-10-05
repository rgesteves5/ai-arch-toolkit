"""20 — Rich Streaming Events.

stream_events() yields structured StreamEvent objects instead of plain
text chunks. Each event has a kind: "text", "thinking", "tool_call_delta" (a piece of a tool
call while the model writes it), "tool_call" (the finished call), or "image".

This is useful for building UIs that render different event types
differently (e.g. collapsible thinking blocks, formatted tool calls).
"""

from ai_arch_toolkit import LLM

llm = LLM("gpt-4.1-mini")

print("=== Rich streaming events ===\n")

stream = llm.stream_events_sync("Explain why the sky is blue in two sentences.")

writing: set[int] = set()  # the tool calls whose pieces have started
for event in stream:
    if event.kind == "text":
        print(event.text, end="", flush=True)
    elif event.kind == "thinking":
        print(f"\n[thinking] {event.thinking.text[:80]}...")
    elif event.kind == "tool_call_delta":
        piece = event.tool_call_delta
        if piece.index not in writing:
            writing.add(piece.index)
            print(f"\n[writing {piece.name}] ", end="")
        print(piece.input_json, end="", flush=True)
    elif event.kind == "tool_call":
        print(f"\n[tool_call] {event.tool_call.name}({event.tool_call.input})")

print(
    f"\n\n[Tokens — in: {stream.response.usage.input_tokens}, "
    f"out: {stream.response.usage.output_tokens}]"
)
print(f"[Cost: ${stream.response.cost:.6f}]")
