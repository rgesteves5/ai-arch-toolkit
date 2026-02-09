"""03 — Streaming (Gemini).

Two streaming modes:
  1. stream()        — yields plain text chunks
  2. stream_events() — yields StreamEvent objects with type metadata
"""

from ai_arch_toolkit import Client

client = Client("gemini", model="gemini-2.0-flash")

# --- Mode 1: Simple text streaming ---
print("=== stream() ===")
for chunk in client.stream("Explain photosynthesis in three sentences."):
    print(chunk, end="", flush=True)
print("\n")

# --- Mode 2: Rich event streaming ---
print("=== stream_events() ===")
for event in client.stream_events("Why is the sky blue? One paragraph."):
    if event.type == "text":
        print(event.text, end="", flush=True)
    elif event.type == "usage":
        print(f"\n\n[Tokens — in: {event.usage.input_tokens}, out: {event.usage.output_tokens}]")
    elif event.type == "done":
        print("[Stream complete]")
