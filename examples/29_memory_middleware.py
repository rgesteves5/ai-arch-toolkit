"""Memory middleware — automatic inject and record on every LLM call.

Demonstrates:
  - Embedding function for vector similarity search
  - MemoryMiddleware wired to SimilarityView (find) and TemporalView (record)
  - Memories auto-injected into system prompt before each LLM call
  - Interactions auto-recorded after each LLM call
  - Cognitive preset for human-inspired memory organization

Requires: OPENAI_API_KEY (or change the model to your provider).
"""

from __future__ import annotations

import asyncio

from ai_arch_toolkit import LLM
from ai_arch_toolkit.toolkit.memory import (
    GraphStore,
    MemoryMiddleware,
    Node,
    SimilarityView,
    TemporalView,
    cognitive,
)
from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend


def make_embed_fn():
    """Create a simple hash-based embedding function.

    For a real app, replace with an embedding API (OpenAI, Cohere, etc.).
    This stand-in produces deterministic 64-dim vectors from text hashes.
    """

    async def embed(text: str) -> list[float]:
        h = hash(text)
        return [(((h >> i) & 0xFF) - 128) / 128.0 for i in range(64)]

    return embed


async def main() -> None:
    print("=== Memory Middleware ===\n")

    embed = make_embed_fn()

    # --- 1. Create store with embeddings ---
    store = GraphStore(NetworkXBackend(), embed=embed)

    # --- 2. Pre-seed some memories ---
    seeds = [
        Node(type="fact", content={"text": "The user's name is Alice"}, source="user_stated"),
        Node(
            type="preference",
            content={"text": "The user prefers concise answers"},
            source="user_stated",
        ),
        Node(
            type="fact",
            content={"text": "The user works at a startup building AI tools"},
            source="user_stated",
        ),
        Node(
            type="fact",
            content={"text": "The user's favorite language is Python"},
            source="user_stated",
        ),
    ]
    for node in seeds:
        await store.add(node)
    print(f"Pre-seeded {await store.count()} memories\n")

    # --- 3. Set up views ---
    similarity = SimilarityView(store, node_type="fact")
    history = TemporalView(store, node_type="interaction")

    # --- 4. Create middleware ---
    middleware = MemoryMiddleware(
        find=similarity.find,
        record=history.append,
        k=3,
        header="What you know about this user:",
    )

    # --- 5. Use LLM with middleware ---
    llm_with_memory = LLM("gpt-4.1-nano", middleware=[middleware])

    questions = [
        "What programming language should I use for my next project?",
        "Can you remind me where I work?",
        "How should you format your responses for me?",
    ]

    for q in questions:
        print(f"User: {q}")
        response = await llm_with_memory.complete(q)
        print(f"Assistant: {response.text[:200]}\n")

    # --- 6. Check what was recorded ---
    print("--- Recorded Interactions ---")
    interactions = await history.recent(k=5)
    for node in interactions:
        query = node.content.get("query", "")[:60]
        resp = node.content.get("response_summary", "")[:60]
        print(f"  Q: {query}...")
        print(f"  A: {resp}...")
        print()

    # --- 7. Show cognitive preset ---
    print("--- Cognitive Preset ---")
    preset = cognitive(store)
    for name in preset.views:
        print(f"  View: {name}")

    # Semantic search across facts
    semantic = preset["semantic"]
    results = await semantic.find("programming language")
    print(f"\n  Semantic search 'programming language': {len(results)} results")
    for r in results:
        text = " ".join(str(v) for v in r.node.content.values() if isinstance(v, str))
        print(f"    [{r.score:.2f}] {text[:70]}")


if __name__ == "__main__":
    asyncio.run(main())
