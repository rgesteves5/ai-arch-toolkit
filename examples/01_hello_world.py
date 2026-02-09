"""01 — Hello World (OpenAI).

Minimal starting point: create a Client, send a single prompt, and
inspect the response text and token usage.
"""

from ai_arch_toolkit import Client

client = Client("openai", model="gpt-5-nano")

response = client.chat("What is the capital of France? Reply in one sentence.")

print("Response:", response.text)
print(f"Tokens — in: {response.usage.input_tokens}, out: {response.usage.output_tokens}")
