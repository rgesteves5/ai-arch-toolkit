"""22 — Retry + Timeout Controls (OpenAI).

Shows how to:
  - configure automatic retries via ``RetryConfig``
  - set per-request ``timeout``
  - handle ``RateLimitError`` and ``APIError``
"""

from ai_arch_toolkit import APIError, Client, RateLimitError, RetryConfig

retry = RetryConfig(
    max_retries=2,
    backoff_factor=1.5,
    retryable_codes=frozenset({429, 500, 502, 503, 504}),
)
client = Client("openai", model="gpt-5-nano", retry=retry)

print("=== Request with retry policy + explicit timeout ===")
try:
    response = client.chat(
        "Give me one sentence explaining what HTTP retries are.",
        timeout=20,
    )
    print("Response:", response.text)
except RateLimitError as exc:
    print(f"Rate limit encountered: status={exc.status_code}, retry_after={exc.retry_after}")
except APIError as exc:
    print(f"API error encountered: status={exc.status_code}")

print("\n=== Intentional tiny timeout to demonstrate timeout handling ===")
try:
    _ = client.chat(
        "Write 3 short bullet points about distributed systems.",
        timeout=0.001,
    )
    print("Call unexpectedly succeeded with tiny timeout.")
except TimeoutError as exc:
    print(f"TimeoutError: {exc}")
except Exception as exc:  # provider/network client timeouts can surface as different exception types
    print(f"Timeout-like failure ({type(exc).__name__}): {exc}")
