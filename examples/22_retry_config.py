"""22 — Retry Configuration.

Configure automatic retries with exponential backoff for transient API
failures (rate limits, 5xx errors). The LLM retries transparently —
you don't need to write retry loops yourself.
"""

from ai_arch_toolkit import LLM
from ai_arch_toolkit.core import RetryConfig

llm = LLM(
    "gpt-4.1-nano",
    retry=RetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        retry_on_status=(429, 500, 502, 503, 504),
    ),
)

result = llm.complete_sync("What is the capital of Japan?")
print("Answer:", result.text)
print(f"Tokens: {result.usage.input_tokens + result.usage.output_tokens}")
