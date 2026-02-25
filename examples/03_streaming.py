"""03 — Streaming.

Stream text chunks from the model. After the stream is consumed,
access the full Response (with usage, cost, etc.) via stream.response.
"""

from ai_arch_toolkit import LLM

llm = LLM("gpt-4.1-nano")

# --- Sync streaming ---
print("=== stream_sync() ===")
stream = llm.stream_sync("Explain photosynthesis in three sentences.")
for chunk in stream:
    print(chunk, end="", flush=True)

# After the stream is fully consumed, .response is available
print(
    f"\n\n[Tokens — in: {stream.response.usage.input_tokens}, "
    f"out: {stream.response.usage.output_tokens}]"
)
print(f"[Cost: ${stream.response.cost:.6f}]")
