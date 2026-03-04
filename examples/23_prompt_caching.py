"""23 — Prompt Caching (Anthropic).

Use cache() to mark text for Anthropic's prompt caching. Cached content
is reused across requests, reducing latency and cost for repeated context.

Requires an Anthropic model (claude-*).
"""

from ai_arch_toolkit import LLM
from ai_arch_toolkit.core import cache, user

llm = LLM("claude-sonnet-4-20250514")

# A long system context that benefits from caching
long_context = (
    "You are an expert on the Python programming language. "
    "Here is a comprehensive reference:\n\n"
    + "Python was created by Guido van Rossum and first released in 1991. "
    * 50  # Repeat to create a large cacheable block
)

# Wrap the context in cache() so Anthropic caches it
messages = [user(["Answer briefly.", cache(long_context), "What is a decorator?"])]

result = llm.complete_sync(messages)
print("Answer:", result.text)
print(f"Cache write tokens: {result.usage.cache_write_tokens}")
print(f"Cache read tokens: {result.usage.cache_read_tokens}")
