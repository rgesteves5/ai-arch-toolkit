"""19 — Guardrails Middleware (OpenAI).

Block disallowed input/output patterns before returning results.
"""

from ai_arch_toolkit import Client, GuardrailMiddleware, GuardrailViolation

guardrails = GuardrailMiddleware(blocked_patterns=["password", "secret"])

client = Client("openai", model="gpt-5-nano", middleware=[guardrails])

try:
    _ = client.chat("My password is 12345. Please remember it.")
except GuardrailViolation as exc:
    print("Blocked by guardrails:", exc)

try:
    safe = client.chat("Give me three best practices for secure credential storage.")
except GuardrailViolation as exc:
    print("\nOutput was blocked by guardrails:", exc)
else:
    print("\nSafe response:")
    print(safe.text)
