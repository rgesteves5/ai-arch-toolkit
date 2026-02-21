"""26 — Cost Reporting with CostTracker (OpenAI).

Tracks usage/cost across multiple requests and prints a per-model breakdown.
"""

from ai_arch_toolkit import Client, CostTracker, ModelPricing

pricing = {
    "openai:gpt-5-nano": ModelPricing(input_per_million=0.15, output_per_million=0.6),
}
tracker = CostTracker(pricing=pricing)
client = Client("openai", model="gpt-5-nano", middleware=[tracker])

prompts = [
    "Give one concise definition of eventual consistency.",
    "Give one concise definition of idempotency.",
]

for i, prompt in enumerate(prompts, start=1):
    response = client.chat(prompt)
    print(f"Call {i}: {response.text}")

print("\nStreaming call (to include stream-events usage accounting):")
for event in client.stream_events("List 3 HTTP methods and one use-case each."):
    if event.type == "text":
        print(event.text, end="", flush=True)
print("\n")

snapshot = tracker.snapshot()
print("=== Cost Snapshot ===")
print(f"requests: {snapshot.request_count}")
print(f"total_tokens: {snapshot.total_usage.total_tokens}")
print(f"estimated_total_cost_usd: {snapshot.total_cost_usd:.8f}")

print("\nPer-model usage/cost:")
for model_key, usage in snapshot.per_model_usage.items():
    cost = snapshot.per_model_cost_usd.get(model_key, 0.0)
    print(
        f"  {model_key}: input={usage.input_tokens}, output={usage.output_tokens}, "
        f"total={usage.total_tokens}, cost_usd={cost:.8f}"
    )
