# Memory

Graph-backed memory for agents. Built on the [`core/graph/`](graph.md) layer: every memory is a `Node`, relationships are `Edge`s, and a `GraphStore` coordinates the backend, an optional vector index, and an optional embedding function.

The whole API is async-first. All symbols below are re-exported from the top-level package (`from ai_arch_toolkit import GraphStore, Node, ...`) or from `ai_arch_toolkit.toolkit.memory`.

---

## GraphStore

The primary facade. It wraps a graph backend and handles auto-embedding, access tracking, type indexing, and persistence.

```python
from ai_arch_toolkit import GraphStore, Node
from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend

store = GraphStore(NetworkXBackend())   # NetworkX backend (needs the [graph] extra)

# Store a memory
node = await store.add(Node(
    type="fact",
    content={"text": "The capital of France is Paris"},
    source="user",
    confidence=0.95,
))

# Retrieve by id (auto-bumps access_count and last_accessed)
node = await store.get(node.id)

# Search (keyword by default — see "Search and recall" below)
results = await store.search("capital France", k=5)
for r in results:
    print(f"{r.node.content['text']} (score: {r.score:.2f})")
```

Constructor:

```python
GraphStore(
    backend,                # MemoryBackend (e.g. NetworkXBackend())
    *,
    embed=None,             # EmbedFn: async (str) -> list[float], enables vector search
    index=None,             # VectorIndex; defaults to BruteForceIndex when embed is set
)
```

Other store methods: `update()`, `remove()`, `list(type=..., limit=...)`, `count()`, `connect()` / `disconnect()` / `edges()` / `neighbors()`, bulk `add_many()` / `remove_many()`, and persistence (`save()` / `load()` / `to_dict()` / `from_dict()`).

---

## Memory Node fields

A memory `Node` is a `core.graph.Node[dict[str, Any]]` with the bookkeeping memory needs. Only **string values** in `content` are keyword-searchable.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | auto (16-char hex) | Unique identifier |
| `type` | `str` | `"generic"` | Node type, used for filtering and indexing |
| `content` | `dict[str, Any]` | `{}` | Searchable key-value payload (string values only) |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata |
| `embedding` | `list[float] \| None` | `None` | Vector embedding (for similarity search) |
| `timestamp` | `datetime` | now (UTC) | Effective / event time |
| `created_at` | `datetime` | now (UTC) | Insertion time |
| `access_count` | `int` | `0` | Bumped on every `get()` |
| `last_accessed` | `datetime \| None` | `None` | Updated on every `get()` |
| `confidence` | `float` | `1.0` | 0.0–1.0 |
| `source` | `str` | `"unknown"` | Provenance |

`timestamp` and `created_at` are tracked separately (bi-temporal): event time vs. insertion time.

---

## Search and recall

`store.search()` returns `SearchResult` objects (a `node` plus a `score`). It runs a cascade:

1. **If an `embed` function is configured** — embed the query, then try the backend's native vector search; fall back to the `VectorIndex` (`BruteForceIndex` by default, cosine similarity).
2. **Otherwise** — keyword search over the string values in `content`.

```python
results = await store.search("capital France", type="fact", k=5)
```

By default (no `embed` wired in), recall is **keyword-only**. To get semantic similarity, enable embeddings.

### Enabling vector similarity

Pass an async embedding function when constructing the store. A `BruteForceIndex` is created automatically (cosine, O(n) — fine for small/medium stores; swap in a custom `VectorIndex` for scale).

```python
async def embed(text: str) -> list[float]:
    # Call your embedding model/provider here.
    ...

store = GraphStore(NetworkXBackend(), embed=embed)

# Nodes are auto-embedded on add(); search now ranks by cosine similarity.
await store.add(Node(type="fact", content={"text": "Paris is the capital of France"}))
results = await store.search("French capital city", k=5)
```

`store.has_embeddings` reports whether an embed function is active.

---

## Views

Views are composable lenses over the store, each optionally scoped to a `node_type`.

```python
from ai_arch_toolkit import TemporalView, RelationalView, PropertyView, SimilarityView
```

### TemporalView — time-based queries and sequential writes

```python
temporal = TemporalView(store, node_type="event")

recent = await temporal.recent(k=10)                   # k most recent by timestamp
last_hour = await temporal.since(hours=1)              # relative offset (hours/minutes)
window = await temporal.between(start_dt, end_dt)      # absolute datetime range
node = await temporal.append({"text": "..."}, link_previous=True)  # writes a NEXT edge
```

### SimilarityView — vector similarity

```python
similarity = SimilarityView(store, node_type="fact")

hits = await similarity.find("query text", k=5)        # SearchResult list
related = await similarity.similar_to(node_id, k=5)    # nearest to an existing node
```

### RelationalView — graph traversal

```python
relational = RelationalView(store)

neighbors = await relational.neighbors(node_id, depth=1, relation="NEXT")
path = await relational.path(from_id, to_id)           # shortest path (needs algorithms backend)
edges = await relational.edges(node_id, direction="out")
await relational.connect(a_id, b_id, "RELATES_TO")
await relational.disconnect(a_id, b_id, "RELATES_TO")
```

### PropertyView — metadata and lifecycle

```python
props = PropertyView(store)

trusted = await props.by_confidence(min_confidence=0.8)
from_user = await props.by_source("user")
most_used = await props.most_accessed(k=5)
least_used = await props.least_accessed(k=5)
tagged = await props.filter(topic="geography")          # exact metadata match
```

### composite_score — blended ranking

Combine similarity, recency (exponential decay), and importance (log-normalized access count) into one score — useful for re-ranking recall results.

```python
from ai_arch_toolkit.toolkit.memory import composite_score

ranked = sorted(
    await similarity.find("query", k=20),
    key=lambda r: composite_score(
        r,
        similarity_weight=0.5,
        recency_weight=0.3,
        importance_weight=0.2,
        recency_half_life_hours=168,   # 1 week
    ),
    reverse=True,
)
```

---

## Presets

A `MemoryPreset` bundles named views over a store and adds a `consolidate()` helper (dedups nodes by content key). Both factories take a `GraphStore` and return a `MemoryPreset`.

```python
from ai_arch_toolkit.toolkit.memory import conversational, cognitive

# Conversational: history / preferences / knowledge
chat = conversational(store)
recent = await chat["history"].recent(k=10)
prefs = await chat["preferences"].by_source("user")

# Cognitive: semantic / episodic / procedural / relations / properties
brain = cognitive(store)
facts = await brain["semantic"].find("...", k=5)
events = await brain["episodic"].recent(k=5)

removed = await chat.consolidate()   # drop duplicate nodes; returns count removed
```

| Preset | Views |
|--------|-------|
| `conversational` | `history` (Temporal/`interaction`), `preferences` (Property/`preference`), `knowledge` (Similarity/`fact`) |
| `cognitive` | `semantic` (Similarity/`fact`), `episodic` (Temporal/`event`), `procedural` (Similarity/`rule`), `relations` (Relational), `properties` (Property) |

---

## memory_tools

Generate `@tool`-decorated functions so an agent can manage its own memory. Returns a `ToolGroup` containing four tools: `remember`, `recall`, `explore_memory`, and `forget_memory`.

```python
from ai_arch_toolkit.toolkit.memory import memory_tools
from ai_arch_toolkit import ToolGroup

mem = memory_tools(store)                  # ToolGroup with 4 tools
tools = ToolGroup(*mem.tools, get_weather) # combine with other tools

flow = react_flow(llm, tools)
# The agent can now call: remember(...), recall("what did we discuss?"),
# explore_memory(node_id), forget_memory(node_id)
```

---

## MemoryMiddleware

Auto-injects relevant memories into the system prompt before each LLM call and records the interaction afterward. It is wired with two callables — a `find` (recall) function and a `record` (write) function — typically taken from views.

```python
from ai_arch_toolkit import MemoryMiddleware, SimilarityView, TemporalView

knowledge = SimilarityView(store, node_type="fact")
history = TemporalView(store, node_type="interaction")

mw = MemoryMiddleware(
    find=knowledge.find,        # async (query, k=...) -> Sequence[SearchResult]
    record=history.append,      # async (content_dict) -> Node
    k=3,
    header="Relevant memories:",
)
llm = LLM("claude-sonnet-4-20250514", middleware=[mw])
# Every call now gets the top-k memories prepended to the system prompt,
# and each turn is recorded back into the store.
```

> Injection happens on the **async** path (`abefore` / `aafter`); the sync hooks are no-ops.

---

## Persistence

`save` and `to_dict` are instance methods; `load` and `from_dict` are classmethods that build a **fresh** store on a given backend (so pass the backend, and optionally `embed=`/`index=`).

```python
from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend

await store.save("memory.json")                                    # dump to disk (atomic, versioned)
restored = await GraphStore.load("memory.json", NetworkXBackend())  # → new GraphStore

data = await store.to_dict()                                       # serialize to a dict
restored = await GraphStore.from_dict(data, NetworkXBackend())      # → new GraphStore
```

---

See [Agents & Capabilities](agents-and-capabilities.md) for a worked "research agent with memory" example, and the [Graph Layer](graph.md) for the backend, edges, and algorithms underneath.
