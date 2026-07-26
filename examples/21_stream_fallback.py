"""21 — Stream Fallback.

When a primary provider fails during streaming, the LLM automatically
falls back to a secondary provider. This works for both stream() and
stream_events().

Requires API keys for both providers (e.g. ANTHROPIC_API_KEY + OPENAI_API_KEY).
"""

from ai_arch_toolkit import LLM

# Primary: Anthropic, Fallback: OpenAI
llm = LLM("claude-sonnet-5", fallback="gpt-4.1-nano")

print("=== Streaming with fallback ===\n")

# If Anthropic fails, OpenAI takes over transparently
stream = llm.stream_sync("What are the three laws of thermodynamics?")
for chunk in stream:
    print(chunk, end="", flush=True)

print("\n\n[Model used: check response for details]")
print(
    f"[Tokens — in: {stream.response.usage.input_tokens}, "
    f"out: {stream.response.usage.output_tokens}]"
)
print(f"[Cost: ${stream.response.cost:.6f}]")
