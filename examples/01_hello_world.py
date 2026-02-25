"""01 — Hello World.

Minimal starting point: create an LLM, send a single prompt, and
inspect the response text and token usage.
"""

from ai_arch_toolkit import LLM

llm = LLM("gpt-4.1-nano")

response = llm.complete_sync("What is the capital of France? Reply in one sentence.")

print("Response:", response.text)
print(f"Tokens — in: {response.usage.input_tokens}, out: {response.usage.output_tokens}")
