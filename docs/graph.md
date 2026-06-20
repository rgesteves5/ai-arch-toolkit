# Graph Layer

The general-purpose graph layer lives in `core/graph/`. It provides typed nodes, directed edges, pluggable backends, and graph algorithms. The Memory system (`toolkit/memory/`) builds on top of this.

---

## Data Structures

### Node[T]

Generic typed node. Content defaults to `None` for type-only marker nodes.

```python
from ai_arch_toolkit.core.graph import Node

# String content
node = Node(type="person", content="Alice")

# Dict content
node = Node(type="fact", content={"text": "Paris is the capital of France"})

# Custom dataclass content
node = Node(type="entity", content=MyDataclass(...))

# Marker node (no content)
node = Node(type="tag")
```

Fields (frozen dataclass):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | auto (16-char hex) | Unique identifier |
| `type` | `str` | `"default"` | Node type for filtering and indexing |
| `content` | `T` | `None` | Generic typed payload |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary key-value metadata |

### Edge

Directed edge between two nodes.

```python
from ai_arch_toolkit.core.graph import Edge

edge = Edge(source="node_a", target="node_b", relation="knows", weight=0.9)
```

Fields (frozen dataclass):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source` | `str` | required | Source node ID |
| `target` | `str` | required | Target node ID |
| `relation` | `str` | required | Edge label/type |
| `weight` | `float` | `1.0` | Numeric weight |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary key-value metadata |

---

## Graph Facade

The `Graph` class is the primary API. It delegates storage to a `GraphBackend` and optionally exposes algorithms when the backend implements `GraphAlgorithms`.

```python
from ai_arch_toolkit.core.graph import Graph, Node
from ai_arch_toolkit.core.graph._networkx import NetworkXBackend

graph = Graph(NetworkXBackend())
```

All methods are **async-first** with `_sync` wrappers (e.g., `graph.add()` / `graph.add_sync()`).

### Node CRUD

```python
# Add
node = await graph.add(Node(type="person", content="Alice"))

# Get by ID
node = await graph.get(node.id)  # → Node or None

# Update attributes (returns new frozen Node)
updated = await graph.update(node.id, content="Alice Smith")

# Remove
removed = await graph.remove(node.id)  # → bool

# Check existence
exists = await graph.has(node.id)  # → bool

# List nodes (optionally by type)
all_nodes = await graph.list()
people = await graph.list(type="person", limit=10)

# Count
total = await graph.node_count()
n_people = await graph.count(type="person")

# Bulk operations
await graph.add_many([node_a, node_b, node_c])
await graph.remove_many([id_a, id_b])
```

### Type Indexing

`Graph` maintains an in-memory type index. Listing nodes by type is **O(k)** where k is the number of nodes of that type, rather than scanning all nodes.

```python
# These use the type index — fast even for large graphs
people = await graph.list(type="person")
count = await graph.count(type="person")
```

### Edge Operations

```python
# Connect two nodes
edge = await graph.connect("alice", "bob", "knows", weight=0.9)

# Get edges for a node
out_edges = await graph.edges("alice", direction="out")
in_edges = await graph.edges("alice", direction="in")
all_edges = await graph.edges("alice", direction="both")

# Filter by relation
knows_edges = await graph.edges("alice", relation="knows")

# Get edges between two specific nodes
edges = await graph.get_edges_between("alice", "bob")

# List all edges in the graph
all = await graph.list_edges()
all_knows = await graph.list_edges(relation="knows")

# Disconnect
removed = await graph.disconnect("alice", "bob", "knows")

# Counts
degree = await graph.degree("alice")
total_edges = await graph.edge_count()
```

---

## Traversal and Algorithms

### Basic Traversal

```python
# Neighbors (BFS within depth)
neighbors = await graph.neighbors("alice", depth=2)
neighbors = await graph.neighbors("alice", relation="knows")
```

### Graph Algorithms

These require the backend to implement the `GraphAlgorithms` protocol (the default `NetworkXBackend` does). Check with `graph.has_algorithms`.

```python
# Breadth-first search
nodes = await graph.bfs("start_node")
nodes = await graph.bfs("start_node", relation="knows")

# Depth-first search
nodes = await graph.dfs("start_node")

# Shortest path (returns node sequence or None)
path = await graph.shortest_path("alice", "dave")

# All simple paths between two nodes
paths = await graph.find_all_paths("alice", "dave", max_depth=5)

# Ancestors and descendants
ancestors = await graph.get_ancestors("node_id")    # → set[NodeID]
descendants = await graph.get_descendants("node_id") # → set[NodeID]
```

### Analysis

```python
# PageRank scores
scores = await graph.pagerank(alpha=0.85)

# Degree centrality
centrality = await graph.centrality()

# Weakly connected components
components = await graph.connected_components()
```

### Subgraph Extraction

```python
# Extract subgraph with specific nodes
sub = await graph.get_subgraph(["alice", "bob", "charlie"])

# Ego graph (neighborhood of a node)
ego = await graph.get_ego_graph("alice", radius=2)
```

---

## Filter and Stats

```python
# Filter nodes by predicate
important = await graph.filter_nodes(lambda n: n.metadata.get("importance", 0) > 5)

# Filter edges by predicate
strong = await graph.filter_edges(lambda e: e.weight > 0.8)

# Find orphan nodes (degree 0)
orphans = await graph.get_orphan_nodes()

# Graph statistics
stats = await graph.get_stats()
# → {"node_count": 42, "edge_count": 87, "node_types": {"person": 10, ...}, "edge_relations": {"knows": 30, ...}}

# Check if empty
empty = await graph.is_empty()
```

---

## Persistence

### JSON Serialization

```python
# Save to file
await graph.save("my_graph.json")

# Load from file
graph = await Graph.load("my_graph.json", NetworkXBackend())

# Dict round-trip
data = await graph.to_dict()
graph = await Graph.from_dict(data, NetworkXBackend())
```

### Content Loader

When deserializing, use `content_loader` to reconstruct typed content from JSON:

```python
def load_content(raw: Any) -> MyDataclass:
    return MyDataclass(**raw)

graph = await Graph.from_dict(data, backend, content_loader=load_content)
graph = await Graph.load("graph.json", backend, content_loader=load_content)
```

### Deep Copy

```python
copy = await graph.copy()
# or with a specific backend:
copy = await graph.copy(backend=NetworkXBackend())
```

---

## Backend Protocols

### GraphBackend (required)

The minimum interface for graph storage. All methods are async.

```python
class GraphBackend(Protocol):
    async def add_node(self, node: Node[Any]) -> None: ...
    async def get_node(self, node_id: NodeID) -> Node[Any] | None: ...
    async def update_node(self, node_id: NodeID, **attrs: object) -> Node[Any] | None: ...
    async def remove_node(self, node_id: NodeID) -> bool: ...
    async def list_nodes(self, *, type: NodeType | None, limit: int | None) -> Sequence[Node[Any]]: ...
    async def count_nodes(self, *, type: NodeType | None) -> int: ...
    async def add_edge(self, edge: Edge) -> None: ...
    async def get_edges(self, node_id: NodeID, *, direction: Direction, relation: str | None) -> Sequence[Edge]: ...
    async def remove_edge(self, source: NodeID, target: NodeID, relation: str) -> bool: ...
    async def neighbors(self, node_id: NodeID, *, depth: int, relation: str | None) -> Sequence[Node[Any]]: ...
    async def clear(self, *, type: NodeType | None) -> int: ...
```

### GraphAlgorithms (optional)

Extended interface for graph algorithms. When a backend implements this, `Graph` exposes traversal and analysis methods.

```python
class GraphAlgorithms(Protocol):
    async def bfs(self, start: NodeID, *, relation: str | None) -> Sequence[Node[Any]]: ...
    async def dfs(self, start: NodeID, *, relation: str | None) -> Sequence[Node[Any]]: ...
    async def shortest_path(self, source: NodeID, target: NodeID, *, relation: str | None) -> Sequence[Node[Any]] | None: ...
    async def centrality(self, *, relation: str | None) -> dict[NodeID, float]: ...
    async def connected_components(self, *, relation: str | None) -> Sequence[Sequence[NodeID]]: ...
    async def subgraph(self, node_ids: Sequence[NodeID]) -> GraphBackend: ...
    async def find_all_paths(self, source: NodeID, target: NodeID, *, max_depth: int | None) -> Sequence[Sequence[NodeID]]: ...
    async def ancestors(self, node_id: NodeID) -> set[NodeID]: ...
    async def descendants(self, node_id: NodeID) -> set[NodeID]: ...
    async def ego_graph(self, node_id: NodeID, *, radius: int) -> GraphBackend: ...
    async def pagerank(self, *, alpha: float) -> dict[NodeID, float]: ...
```

Check at runtime:

```python
graph.has_algorithms  # True if backend implements GraphAlgorithms
```

---

## NetworkXBackend

The default in-memory implementation. Uses a NetworkX `MultiDiGraph` (supports multiple edges between the same pair of nodes via the `relation` as edge key). Implements both `GraphBackend` and `GraphAlgorithms`.

```python
from ai_arch_toolkit.core.graph._networkx import NetworkXBackend

backend = NetworkXBackend()
graph = Graph(backend)
```

Import-guarded — requires the `[graph]` extra:

```bash
pip install ai-arch-toolkit[graph]
```

The `NetworkXBackend` is not re-exported from the `core.graph` package to keep the import guard effective. Import it directly from `core.graph._networkx`.

---

## Connection to Memory

The Memory system (`toolkit/memory/`) wraps `Graph` with memory-specific features:

```python
from ai_arch_toolkit import GraphStore
from ai_arch_toolkit.core.graph import Graph

graph = Graph()
store = GraphStore(graph.backend)
```

`GraphStore` adds:
- Memory-specific `Node` fields (`timestamp`, `source`, `confidence`, `access_count`, `last_accessed`, `embedding`)
- Keyword and vector search
- Access tracking (automatic `access_count` and `last_accessed` updates on `get()`)
- Views: `TemporalView`, `RelationalView`, `PropertyView`, `SimilarityView`

See [Memory](memory.md) for details.
