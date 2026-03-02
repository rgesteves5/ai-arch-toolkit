"""Graph memory basics — store, search, views, and persistence.

Demonstrates:
  - Creating a GraphStore with an in-memory NetworkX backend
  - Adding nodes of different types (facts, events, rules)
  - Keyword search (no embedding function needed)
  - Temporal, relational, and property views
  - Graph traversal and edge creation
  - Persistence (save/load to JSON)

No API keys required — this example uses keyword search only.
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_arch_toolkit.toolkit.memory import (
    GraphStore,
    Node,
    PropertyView,
    RelationalView,
    TemporalView,
    composite_score,
)
from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend


async def main() -> None:
    # --- 1. Create a store ---
    backend = NetworkXBackend()
    store = GraphStore(backend)
    print("=== Graph Memory Basics ===\n")

    # --- 2. Add nodes of different types ---
    now = datetime.now(UTC)

    facts = [
        Node(id="f1", type="fact", content={"text": "Python was created by Guido van Rossum"}),
        Node(id="f2", type="fact", content={"text": "Python 3.13 added type parameter syntax"}),
        Node(id="f3", type="fact", content={"text": "Rust is a systems programming language"}),
    ]
    events = [
        Node(
            id="e1",
            type="event",
            content={"text": "User asked about Python history"},
            timestamp=now - timedelta(hours=2),
            source="user_stated",
        ),
        Node(
            id="e2",
            type="event",
            content={"text": "User asked about type hints"},
            timestamp=now - timedelta(minutes=30),
            source="user_stated",
        ),
    ]
    rules = [
        Node(
            id="r1",
            type="rule",
            content={"condition": "user asks about Python", "action": "check Python facts first"},
            confidence=0.9,
        ),
    ]

    for node in [*facts, *events, *rules]:
        await store.add(node)

    print(f"Added {await store.count()} nodes")
    print(f"  Facts: {await store.count(type='fact')}")
    print(f"  Events: {await store.count(type='event')}")
    print(f"  Rules: {await store.count(type='rule')}")

    # --- 3. Keyword search (no embeddings needed) ---
    print("\n--- Keyword Search ---")
    results = await store.search("Python")
    for r in results:
        text = " ".join(str(v) for v in r.node.content.values() if isinstance(v, str))
        print(f"  [{r.node.type}] {text[:70]}")

    # --- 4. Connect nodes with edges ---
    print("\n--- Building Relations ---")
    await store.connect("f1", "f2", "RELATED_TO", metadata={"reason": "same language"})
    await store.connect("e1", "f1", "TRIGGERED_LOOKUP")
    await store.connect("e2", "f2", "TRIGGERED_LOOKUP")
    print("  f1 --RELATED_TO--> f2")
    print("  e1 --TRIGGERED_LOOKUP--> f1")
    print("  e2 --TRIGGERED_LOOKUP--> f2")

    # --- 5. Temporal view ---
    print("\n--- Temporal View (events) ---")
    temporal = TemporalView(store, node_type="event")
    recent = await temporal.recent(k=5)
    for node in recent:
        age = now - node.timestamp
        print(f"  [{node.id}] {node.content['text']} ({age.seconds // 60}m ago)")

    only_recent = await temporal.since(hours=1)
    print(f"  Events in last hour: {len(only_recent)}")

    # --- 6. Relational view ---
    print("\n--- Relational View ---")
    relational = RelationalView(store)
    neighbors = await relational.neighbors("f1", depth=1)
    print(f"  Neighbors of f1: {[n.id for n in neighbors]}")

    path = await relational.path("e1", "f2")
    if path:
        print(f"  Path e1 → f2: {' → '.join(n.id for n in path)}")

    # --- 7. Property view ---
    print("\n--- Property View ---")
    props = PropertyView(store)

    high_confidence = await props.by_confidence(min_confidence=0.8)
    print(f"  High confidence (≥0.8): {len(high_confidence)} nodes")

    user_stated = await props.by_source("user_stated")
    print(f"  User-stated: {len(user_stated)} nodes")

    # --- 8. Access tracking ---
    print("\n--- Access Tracking ---")
    node = await store.get("f1")  # bumps access count
    node = await store.get("f1")  # bumps again
    node = await store.get("f1")  # and again
    print(f"  f1 accessed {node.access_count}x, last at {node.last_accessed:%H:%M:%S}")

    most = await props.most_accessed(k=2)
    print(f"  Most accessed: {[(n.id, n.access_count) for n in most]}")

    # --- 9. Composite scoring ---
    print("\n--- Composite Scoring ---")
    results = await store.search("Python")
    for r in results:
        score = composite_score(r, recency_half_life_hours=24)
        text = " ".join(str(v) for v in r.node.content.values() if isinstance(v, str))
        print(f"  score={score:.3f} [{r.node.type}] {text[:60]}")

    # --- 10. Persistence ---
    print("\n--- Persistence ---")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "memory.json"
        await store.save(path)
        print(f"  Saved to {path.name} ({path.stat().st_size} bytes)")

        loaded = await GraphStore.load(path, NetworkXBackend())
        print(f"  Loaded {await loaded.count()} nodes")
        edges = await loaded.edges("f1")
        print(f"  f1 edges preserved: {len(edges)}")


if __name__ == "__main__":
    asyncio.run(main())
