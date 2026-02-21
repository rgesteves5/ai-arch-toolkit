"""25 — Custom Cache Backend (OpenAI).

Implements a custom ``CacheBackend`` and wires it into ``ResponseCache``.
"""

from __future__ import annotations

from ai_arch_toolkit import CacheBackend, Client, Response, ResponseCache


class CountingCacheBackend(CacheBackend):
    """In-memory cache backend that also tracks hit/miss/set metrics."""

    def __init__(self) -> None:
        self._values: dict[str, Response] = {}
        self.hits = 0
        self.misses = 0
        self.sets = 0

    def get(self, key: str) -> Response | None:
        value = self._values.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return value

    def set(self, key: str, value: Response, ttl_seconds: float | None) -> None:
        self.sets += 1
        self._values[key] = value


backend = CountingCacheBackend()
cache = ResponseCache(backend=backend, ttl_seconds=300)
client = Client("openai", model="gpt-5-nano", middleware=[cache])

prompt = "Name one design pattern and explain it in one sentence."

print("=== First call (cache miss expected) ===")
first = client.chat(prompt)
print(first.text)

print("\n=== Second call (cache hit expected) ===")
second = client.chat(prompt)
print(second.text)

print("\nCache backend metrics:")
print(f"  hits: {backend.hits}")
print(f"  misses: {backend.misses}")
print(f"  sets: {backend.sets}")
print(f"  same_text: {first.text == second.text}")
