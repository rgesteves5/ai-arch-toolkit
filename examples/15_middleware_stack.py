"""15 — Middleware Stack (OpenAI).

Compose multiple middleware in one client:
  - ResponseCache
  - CostTracker
  - GuardrailMiddleware
  - TracingMiddleware
"""

from ai_arch_toolkit import (
    Client,
    CostTracker,
    GuardrailMiddleware,
    ModelPricing,
    ResponseCache,
    TracingMiddleware,
)


class _PrintSpan:
    def set_attribute(self, key, value):
        print(f"[trace] {key}={value}")

    def record_exception(self, exception):
        print(f"[trace] exception={type(exception).__name__}: {exception}")


class _PrintSpanCM:
    def __init__(self, name):
        self._name = name
        self._span = _PrintSpan()

    def __enter__(self):
        print(f"[trace] start span={self._name}")
        return self._span

    def __exit__(self, exc_type, exc, tb):
        print(f"[trace] end span={self._name}")


class _PrintTracer:
    def start_as_current_span(self, name):
        return _PrintSpanCM(name)


cache = ResponseCache(ttl_seconds=300)
cost = CostTracker(
    pricing={
        "openai:gpt-5-nano": ModelPricing(input_per_million=0.15, output_per_million=0.6),
    }
)
guardrails = GuardrailMiddleware(blocked_patterns=["password", "secret"])
tracing = TracingMiddleware(tracer=_PrintTracer())

client = Client(
    "openai",
    model="gpt-5-nano",
    middleware=[cache, cost, guardrails, tracing],
)

question = "Explain the observer effect in quantum physics in two short paragraphs."

print("=== First call (cache miss) ===")
resp1 = client.chat(question)
print(resp1.text[:400], "...\n")

print("=== Second call (cache hit) ===")
resp2 = client.chat(question)
print(resp2.text[:200], "...\n")

snapshot = cost.snapshot()
print("=== Cost Snapshot ===")
print(f"requests: {snapshot.request_count}")
print(f"total tokens: {snapshot.total_usage.total_tokens}")
print(f"estimated cost (USD): {snapshot.total_cost_usd:.8f}")
