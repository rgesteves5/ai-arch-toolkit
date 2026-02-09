# NetworkX Guide

> Practical reference for graph creation, algorithms, and analysis with NetworkX.
> NetworkX 3.6+ · Python 3.12+ · Last updated: February 2026

**Official docs:** [Reference](https://networkx.org/documentation/stable/reference/index.html) · [Tutorial](https://networkx.org/documentation/stable/tutorial.html) · [Guides](https://networkx.org/nx-guides/)

---

## Table of Contents

1. [Setup](#1-setup)
2. [Graph Types — Which One to Use](#2-graph-types--which-one-to-use)
3. [Graph Theory — Types of Graphs](#3-graph-theory--types-of-graphs)
4. [Creating Graphs](#4-creating-graphs)
5. [Nodes & Edges — Core Operations](#5-nodes--edges--core-operations)
6. [Attributes](#6-attributes)
7. [Querying & Inspecting](#7-querying--inspecting)
8. [Algorithms Quick Reference](#8-algorithms-quick-reference)
9. [Graph Generators](#9-graph-generators)
10. [Import & Export](#10-import--export)
11. [Visualization](#11-visualization)
12. [Conversion — pandas, numpy, scipy](#12-conversion--pandas-numpy-scipy)
13. [Performance & Scaling](#13-performance--scaling)
14. [Common Recipes](#14-common-recipes)
15. [Common Pitfalls](#15-common-pitfalls)

---

## 1. Setup

```bash
uv add networkx               # core (no optional deps)
uv add "networkx[default]"    # + numpy, scipy, matplotlib, pandas
uv add "networkx[extra]"      # + lxml, pygraphviz, pydot, sympy
```

```python
import networkx as nx    # always use this alias
```

---

## 2. Graph Types — Which One to Use

NetworkX provides four graph classes. Pick based on two questions:
**Are edges directed?** and **Can there be multiple edges between the same pair of nodes?**

```
                    Single edge          Multiple edges
                 ┌─────────────────┬──────────────────────┐
  Undirected     │   nx.Graph()    │   nx.MultiGraph()    │
                 ├─────────────────┼──────────────────────┤
  Directed       │   nx.DiGraph()  │   nx.MultiDiGraph()  │
                 └─────────────────┴──────────────────────┘
```

### `nx.Graph()` — Undirected, single edge

The default. Use for most general-purpose graphs.

```python
G = nx.Graph()
G.add_edge("A", "B")
G.add_edge("B", "A")    # same edge — ignored (already exists)
print(G.number_of_edges())  # 1
```

**When to use:** friendships, undirected road networks, co-authorship, similarity graphs, any symmetric relationship.

### `nx.DiGraph()` — Directed, single edge

Edges have direction: `(u, v)` ≠ `(v, u)`.

```python
DG = nx.DiGraph()
DG.add_edge("A", "B")   # A → B
DG.add_edge("B", "A")   # B → A (different edge)
print(DG.number_of_edges())  # 2

# Direction-aware methods
DG.in_degree("B")        # edges coming IN to B
DG.out_degree("B")       # edges going OUT from B
DG.predecessors("B")     # nodes with edges TO B
DG.successors("B")       # nodes B points TO
```

**When to use:** web links, citations, dependencies (package/task), workflows, follower graphs, causal models, DAGs.

### `nx.MultiGraph()` — Undirected, multiple edges

Allows parallel edges between the same pair of nodes. Each edge gets a unique key.

```python
MG = nx.MultiGraph()
MG.add_edge("A", "B", key="bus", route="Line 1")
MG.add_edge("A", "B", key="train", route="Express")
print(MG.number_of_edges())  # 2

# Access specific edges
MG.edges["A", "B", "bus"]    # {'route': 'Line 1'}
```

**When to use:** transportation networks (multiple routes), communication networks (multiple channels), any relationship that can occur more than once between the same pair.

### `nx.MultiDiGraph()` — Directed, multiple edges

Directed + parallel edges.

```python
MDG = nx.MultiDiGraph()
MDG.add_edge("A", "B", type="email")
MDG.add_edge("A", "B", type="slack")
MDG.add_edge("B", "A", type="email")
print(MDG.number_of_edges())  # 3
```

**When to use:** multi-relational data, knowledge graphs, multi-modal networks.

### Conversion between types

```python
# Directed → Undirected (lose direction info)
UG = DG.to_undirected()

# Undirected → Directed (each edge becomes two directed edges)
DG = G.to_directed()

# Multi → Simple (collapse parallel edges — pick one)
simple = nx.Graph(MG)

# Copy a graph
G2 = G.copy()           # independent copy
```

### Self-loops

All four types allow self-loops:

```python
G.add_edge("A", "A")    # self-loop on node A
nx.number_of_selfloops(G)
list(nx.selfloop_edges(G))
```

### Quick decision guide

```
Do edges have direction?
├─ No  → Are there parallel edges?
│        ├─ No  → nx.Graph()          (most common)
│        └─ Yes → nx.MultiGraph()
└─ Yes → Are there parallel edges?
         ├─ No  → nx.DiGraph()        (second most common)
         └─ Yes → nx.MultiDiGraph()
```

> **Note:** Many algorithms only work on `Graph` or `DiGraph`. MultiGraph users often need to convert to a simple graph first. Check the docs for each algorithm.

---

## 3. Graph Theory — Types of Graphs

Beyond NetworkX classes, these are the conceptual graph types you'll encounter. NetworkX can represent and detect all of them.

### By edge weight

| Type | Description | NetworkX |
|------|-------------|----------|
| **Unweighted** | All edges equal | Default — no `weight` attribute |
| **Weighted** | Edges have numeric cost/distance | `G.add_edge(u, v, weight=3.5)` |

### By structure

| Type | Description | How to check / create |
|------|-------------|----------------------|
| **Tree** | Connected, no cycles, n-1 edges | `nx.is_tree(G)` |
| **Forest** | Collection of trees (disconnected) | `nx.is_forest(G)` |
| **DAG** (directed acyclic graph) | Directed, no cycles | `nx.is_directed_acyclic_graph(G)` |
| **Bipartite** | Nodes split into two disjoint sets, edges only between sets | `nx.is_bipartite(G)` |
| **Complete** | Every pair of nodes connected | `nx.complete_graph(n)` |
| **Planar** | Can be drawn on a plane without edge crossings | `nx.is_planar(G)` |
| **Connected** | Path exists between every pair of nodes | `nx.is_connected(G)` |
| **Strongly connected** (directed) | Directed path between every pair in both directions | `nx.is_strongly_connected(DG)` |
| **Weakly connected** (directed) | Connected if you ignore edge direction | `nx.is_weakly_connected(DG)` |
| **Eulerian** | Has a cycle visiting every edge exactly once | `nx.is_eulerian(G)` |
| **Regular** | Every node has the same degree | `nx.is_regular(G)` |

### By domain / purpose

| Type | Typical use case | Graph class |
|------|-----------------|-------------|
| **Social network** | Friendships, followers | `Graph` or `DiGraph` |
| **Knowledge graph** | Entity-relationship triples | `MultiDiGraph` |
| **Dependency graph** | Package deps, task scheduling | `DiGraph` (usually a DAG) |
| **Flow network** | Supply chains, traffic | `DiGraph` with capacity attributes |
| **Similarity graph** | Document/item similarity | `Graph` with weight = similarity |
| **Road network** | GPS routing | `Graph` or `DiGraph` with weight = distance |

---

## 4. Creating Graphs

### From scratch

```python
G = nx.Graph()

# Add nodes
G.add_node(1)
G.add_node("server-a", role="web")
G.add_nodes_from([2, 3, 4])
G.add_nodes_from([
    (5, {"role": "db"}),
    (6, {"role": "cache"}),
])

# Add edges
G.add_edge(1, 2)
G.add_edge(1, 3, weight=4.2)
G.add_edges_from([(2, 3), (3, 4)])
G.add_weighted_edges_from([
    ("A", "B", 0.5),
    ("B", "C", 1.3),
    ("A", "C", 0.9),
])
```

### From data structures

```python
# From edge list
G = nx.Graph([(1, 2), (2, 3), (3, 1)])

# From adjacency dict
G = nx.Graph({
    "A": ["B", "C"],
    "B": ["A", "D"],
    "C": ["A"],
    "D": ["B"],
})

# From numpy adjacency matrix
import numpy as np
A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
G = nx.from_numpy_array(A)

# From pandas edgelist
import pandas as pd
df = pd.DataFrame({
    "source": ["A", "B", "C"],
    "target": ["B", "C", "A"],
    "weight": [1.0, 2.0, 3.0],
})
G = nx.from_pandas_edgelist(df, "source", "target", edge_attr="weight")

# From pandas adjacency matrix
adj_df = nx.to_pandas_adjacency(G)
G2 = nx.from_pandas_adjacency(adj_df)

# From scipy sparse matrix
from scipy.sparse import csr_array
sparse = nx.to_scipy_sparse_array(G)
G3 = nx.from_scipy_sparse_array(sparse)
```

### From generators (see section 9)

```python
G = nx.complete_graph(5)
G = nx.cycle_graph(10)
G = nx.path_graph(7)
G = nx.star_graph(6)
G = nx.grid_2d_graph(4, 4)
G = nx.erdos_renyi_graph(100, 0.05)
```

---

## 5. Nodes & Edges — Core Operations

### Adding

```python
G.add_node(n)                              # single node
G.add_nodes_from([n1, n2, n3])             # multiple nodes
G.add_edge(u, v)                           # single edge (creates nodes if missing)
G.add_edges_from([(u1, v1), (u2, v2)])     # multiple edges
G.add_weighted_edges_from([(u, v, w)])     # with weight
```

### Removing

```python
G.remove_node(n)                           # node + all its edges
G.remove_nodes_from([n1, n2])
G.remove_edge(u, v)
G.remove_edges_from([(u1, v1)])
G.clear()                                  # remove everything
G.clear_edges()                            # keep nodes, remove edges
```

### Checking existence

```python
n in G                                     # is node in graph?
G.has_node(n)                              # same
G.has_edge(u, v)                           # is edge in graph?
(u, v) in G.edges                          # same
```

### Counting

```python
G.number_of_nodes()     # or len(G)
G.number_of_edges()
G.order()               # same as number_of_nodes
G.size()                # same as number_of_edges
G.size(weight="weight") # sum of all edge weights
```

---

## 6. Attributes

Attributes can be attached to the graph, nodes, and edges.

### Graph attributes

```python
G.graph["name"] = "My Network"
G.graph["created"] = "2026-02-08"
```

### Node attributes

```python
# Set on creation
G.add_node("A", color="red", size=10)

# Set after creation
G.nodes["A"]["color"] = "blue"

# Bulk set
nx.set_node_attributes(G, {"A": "red", "B": "blue"}, name="color")

# Read
G.nodes["A"]                    # {'color': 'blue', 'size': 10}
G.nodes["A"]["color"]           # 'blue'
nx.get_node_attributes(G, "color")  # {'A': 'blue', 'B': 'blue'}

# Iterate with attributes
for node, data in G.nodes(data=True):
    print(node, data)
```

### Edge attributes

```python
# Set on creation
G.add_edge("A", "B", weight=3.5, label="friendship")

# Set after creation
G.edges["A", "B"]["weight"] = 4.0

# Bulk set
nx.set_edge_attributes(G, {("A", "B"): 5.0}, name="weight")

# Read
G.edges["A", "B"]                      # {'weight': 4.0, 'label': 'friendship'}
nx.get_edge_attributes(G, "weight")    # {('A', 'B'): 4.0}

# Iterate with attributes
for u, v, data in G.edges(data=True):
    print(u, v, data)

# Iterate with specific attribute
for u, v, w in G.edges(data="weight", default=1.0):
    print(u, v, w)
```

---

## 7. Querying & Inspecting

### Neighbors & degree

```python
list(G.neighbors("A"))            # adjacent nodes
list(G.adj["A"])                  # same
G.degree("A")                    # number of edges
G.degree("A", weight="weight")   # weighted degree (sum of weights)
dict(G.degree())                 # {node: degree} for all nodes

# DiGraph specific
DG.in_degree("A")
DG.out_degree("A")
list(DG.predecessors("A"))       # nodes with edges TO A
list(DG.successors("A"))         # nodes A points TO
```

### Subgraphs

```python
# Node-induced subgraph (view — reflects changes to original)
sub = G.subgraph(["A", "B", "C"])

# Independent copy
sub = G.subgraph(["A", "B", "C"]).copy()

# Edge subgraph
esub = G.edge_subgraph([("A", "B"), ("B", "C")])

# Ego graph (node + neighbors within radius)
ego = nx.ego_graph(G, "A", radius=2)
```

### Connected components

```python
# Undirected
nx.is_connected(G)
components = list(nx.connected_components(G))            # [{nodes}, {nodes}, ...]
largest = max(nx.connected_components(G), key=len)
G_largest = G.subgraph(largest).copy()
nx.number_connected_components(G)

# Directed
nx.is_strongly_connected(DG)
nx.is_weakly_connected(DG)
list(nx.strongly_connected_components(DG))
list(nx.weakly_connected_components(DG))
```

### Basic statistics

```python
nx.density(G)                       # actual edges / possible edges
nx.average_shortest_path_length(G)  # requires connected graph
nx.diameter(G)                      # longest shortest path
nx.radius(G)                        # min eccentricity
nx.center(G)                        # nodes with eccentricity == radius
nx.average_clustering(G)
nx.transitivity(G)                  # global clustering coefficient
dict(nx.triangles(G))               # triangles per node
```

---

## 8. Algorithms Quick Reference

### Shortest paths

```python
# Unweighted — BFS
nx.shortest_path(G, source="A", target="D")
nx.shortest_path_length(G, source="A", target="D")
nx.all_shortest_paths(G, "A", "D")

# Weighted — Dijkstra (default for weighted)
nx.shortest_path(G, "A", "D", weight="weight")
nx.dijkstra_path(G, "A", "D")
nx.dijkstra_path_length(G, "A", "D")

# Negative weights — Bellman-Ford
nx.bellman_ford_path(G, "A", "D")
nx.bellman_ford_path_length(G, "A", "D")

# All-pairs shortest paths
dict(nx.all_pairs_shortest_path_length(G))       # unweighted
dict(nx.all_pairs_dijkstra_path_length(G))        # weighted

# A* (requires heuristic)
nx.astar_path(G, "A", "D", heuristic=my_heuristic)

# Has path?
nx.has_path(G, "A", "D")
```

### Minimum spanning tree

```python
T = nx.minimum_spanning_tree(G, weight="weight")            # Kruskal's (default)
T = nx.minimum_spanning_tree(G, algorithm="prim")            # Prim's
edges = list(nx.minimum_spanning_edges(G, data=True))
total_weight = T.size(weight="weight")
```

### Centrality

```python
nx.degree_centrality(G)                # {node: centrality}
nx.betweenness_centrality(G)           # nodes on shortest paths
nx.closeness_centrality(G)             # inverse avg distance
nx.eigenvector_centrality(G)           # importance via neighbors
nx.pagerank(DG)                        # PageRank (directed)
nx.katz_centrality(G)                  # Katz centrality

# Approximate betweenness (faster for large graphs)
nx.betweenness_centrality(G, k=100)    # sample k nodes
```

### Community detection

```python
from networkx.algorithms.community import (
    greedy_modularity_communities,
    louvain_communities,
    label_propagation_communities,
    asyn_lpa_communities,
)

communities = louvain_communities(G, seed=42)         # list of sets
communities = greedy_modularity_communities(G)
communities = list(label_propagation_communities(G))

# Modularity score
nx.community.modularity(G, communities)
```

### Traversal

```python
# BFS
list(nx.bfs_edges(G, source="A"))
list(nx.bfs_tree(G, source="A"))
list(nx.bfs_layers(G, sources=["A"]))

# DFS
list(nx.dfs_edges(G, source="A"))
list(nx.dfs_tree(G, source="A"))
list(nx.dfs_preorder_nodes(G, source="A"))
list(nx.dfs_postorder_nodes(G, source="A"))
```

### Cycle detection

```python
# Undirected
nx.cycle_basis(G)                          # list of cycles
list(nx.simple_cycles(DG))                 # all simple cycles (directed)
nx.find_cycle(G)                           # find one cycle (raises if none)

# DAG check
nx.is_directed_acyclic_graph(DG)
```

### Topological sort (DAGs only)

```python
list(nx.topological_sort(DG))              # linear ordering
list(nx.topological_generations(DG))       # grouped by generation/level
list(nx.lexicographical_topological_sort(DG))
```

### Network flow

```python
# Maximum flow
flow_value, flow_dict = nx.maximum_flow(DG, "source", "sink", capacity="capacity")

# Minimum cut
cut_value, partition = nx.minimum_cut(DG, "source", "sink", capacity="capacity")

# Min cost flow
nx.min_cost_flow(DG)
```

### Matching

```python
# Maximum matching (undirected)
matching = nx.max_weight_matching(G)

# Bipartite matching
from networkx.algorithms import bipartite
matching = bipartite.maximum_matching(G, top_nodes)
```

### Coloring

```python
coloring = nx.greedy_color(G, strategy="largest_first")   # {node: color_int}
nx.is_bipartite(G)    # equivalent to 2-colorable
```

### Cliques

```python
list(nx.find_cliques(G))                   # all maximal cliques
nx.graph_clique_number(G)                  # size of largest clique
nx.graph_number_of_cliques(G)              # count of maximal cliques
```

### Link prediction

```python
preds = nx.jaccard_coefficient(G, [("A", "D"), ("B", "E")])
for u, v, score in preds:
    print(u, v, score)

# Other predictors
nx.adamic_adar_index(G, ebunch)
nx.preferential_attachment(G, ebunch)
nx.resource_allocation_index(G, ebunch)
```

### Isomorphism

```python
nx.is_isomorphic(G1, G2)
nx.could_be_isomorphic(G1, G2)             # fast check (necessary conditions)

from networkx.algorithms import isomorphism
GM = isomorphism.GraphMatcher(G1, G2)
GM.is_isomorphic()
GM.subgraph_is_isomorphic()
```

---

## 9. Graph Generators

### Classic graphs

```python
nx.complete_graph(10)                      # K₁₀
nx.complete_bipartite_graph(3, 4)          # K₃,₄
nx.cycle_graph(20)                         # C₂₀
nx.path_graph(15)                          # P₁₅
nx.star_graph(8)                           # S₈ (center + 8 leaves)
nx.wheel_graph(10)                         # cycle + hub
nx.grid_2d_graph(5, 5)                     # 5×5 grid
nx.grid_graph([3, 3, 3])                   # 3D grid
nx.hypercube_graph(4)                      # 4-dimensional hypercube
nx.petersen_graph()
nx.tutte_graph()
```

### Random graphs

```python
# Erdős–Rényi: each edge with probability p
nx.erdos_renyi_graph(n=100, p=0.05, seed=42)
nx.gnp_random_graph(100, 0.05, seed=42)    # same thing

# Barabási–Albert: preferential attachment (scale-free)
nx.barabasi_albert_graph(n=500, m=3, seed=42)

# Watts–Strogatz: small-world
nx.watts_strogatz_graph(n=100, k=4, p=0.3, seed=42)

# Random regular graph
nx.random_regular_graph(d=3, n=50, seed=42)

# Stochastic block model (community structure)
nx.stochastic_block_model(
    sizes=[25, 25, 25],
    p=[[0.5, 0.01, 0.01],
       [0.01, 0.5, 0.01],
       [0.01, 0.01, 0.5]],
    seed=42,
)

# Random tree
nx.random_tree(n=50, seed=42)
```

### Social & benchmark graphs

```python
nx.karate_club_graph()             # Zachary's karate club
nx.les_miserables_graph()          # character co-occurrence
nx.florentine_families_graph()     # marriage/business network
nx.davis_southern_women_graph()    # bipartite
```

---

## 10. Import & Export

### File formats

```python
# GraphML (recommended for interoperability)
nx.write_graphml(G, "graph.graphml")
G = nx.read_graphml("graph.graphml")

# GML
nx.write_gml(G, "graph.gml")
G = nx.read_gml("graph.gml")

# Edge list (simple, human-readable)
nx.write_edgelist(G, "graph.edgelist")
G = nx.read_edgelist("graph.edgelist")
nx.write_weighted_edgelist(G, "graph.weighted.edgelist")
G = nx.read_weighted_edgelist("graph.weighted.edgelist")

# Adjacency list
nx.write_adjlist(G, "graph.adjlist")
G = nx.read_adjlist("graph.adjlist")

# GEXF (for Gephi)
nx.write_gexf(G, "graph.gexf")
G = nx.read_gexf("graph.gexf")

# Pajek
nx.write_pajek(G, "graph.net")
G = nx.read_pajek("graph.net")

# Graph6 / Sparse6 (compact, undirected only)
nx.write_graph6(G, "graph.g6")
G = nx.read_graph6("graph.g6")
```

### JSON (for web apps)

```python
from networkx.readwrite import json_graph
import json

# Node-link format (d3.js compatible)
data = json_graph.node_link_data(G)
json_str = json.dumps(data)
G2 = json_graph.node_link_graph(json.loads(json_str))

# Adjacency format
data = json_graph.adjacency_data(G)

# Tree format (for tree graphs)
data = json_graph.tree_data(G, root="root_node")

# Cytoscape.js format
data = json_graph.cytoscape_data(G)
```

### Choosing a format

| Format | Best for | Attributes? | Human-readable? |
|--------|----------|-------------|-----------------|
| **GraphML** | General interchange, tools like Gephi/yEd | Yes | XML |
| **JSON** (node-link) | Web visualization (d3.js, Cytoscape.js) | Yes | Yes |
| **GEXF** | Gephi import/export | Yes | XML |
| **GML** | General interchange | Yes | Yes |
| **Edge list** | Simple/quick storage | Limited | Yes |
| **Adjacency list** | Compact, no attributes | No | Yes |
| **Graph6/Sparse6** | Compact undirected only | No | No |

---

## 11. Visualization

### Quick plot with matplotlib

```python
import matplotlib.pyplot as plt

# Simplest
nx.draw(G, with_labels=True)
plt.show()

# With layout
pos = nx.spring_layout(G, seed=42)     # force-directed
nx.draw(G, pos, with_labels=True, node_color="lightblue",
        edge_color="gray", node_size=500, font_size=10)
plt.show()
```

### Layout algorithms

```python
pos = nx.spring_layout(G, seed=42)         # force-directed (Fruchterman-Reingold)
pos = nx.kamada_kawai_layout(G)            # force-directed (Kamada-Kawai)
pos = nx.circular_layout(G)               # nodes on a circle
pos = nx.shell_layout(G, nlist=[...])      # concentric circles
pos = nx.spectral_layout(G)               # eigenvalue-based
pos = nx.planar_layout(G)                 # planar (if graph is planar)
pos = nx.random_layout(G, seed=42)        # random positions
pos = nx.bipartite_layout(G, top_nodes)   # two columns

# For trees
pos = nx.bfs_layout(G, start="root")

# Graphviz layouts (requires pygraphviz or pydot)
pos = nx.nx_agraph.graphviz_layout(G, prog="dot")     # hierarchical
pos = nx.nx_agraph.graphviz_layout(G, prog="neato")   # spring
pos = nx.nx_agraph.graphviz_layout(G, prog="circo")   # circular
```

### Custom drawing

```python
pos = nx.spring_layout(G, seed=42)

# Draw nodes with varying size/color
node_sizes = [G.degree(n) * 100 for n in G.nodes()]
node_colors = [G.nodes[n].get("group", 0) for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                        node_color=node_colors, cmap=plt.cm.Set3)

# Draw edges with varying width
edge_widths = [G.edges[e].get("weight", 1) for e in G.edges()]
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=8)
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.axis("off")
plt.tight_layout()
plt.savefig("graph.png", dpi=300, bbox_inches="tight")
plt.show()
```

### Better visualization (for larger/interactive graphs)

For anything beyond simple plots, consider dedicated tools:

```python
# PyVis — interactive HTML graphs
from pyvis.network import Network
net = Network(notebook=True)
net.from_nx(G)
net.show("graph.html")

# Graphviz — publication-quality layouts
A = nx.nx_agraph.to_agraph(G)
A.draw("graph.pdf", prog="dot")
```

---

## 12. Conversion — pandas, numpy, scipy

### To/from pandas

```python
# Edge list DataFrame
df = nx.to_pandas_edgelist(G)             # columns: source, target, [attributes]
G = nx.from_pandas_edgelist(df, "source", "target", edge_attr=True)

# Adjacency matrix DataFrame
adj = nx.to_pandas_adjacency(G, weight="weight")
G = nx.from_pandas_adjacency(adj)
```

### To/from numpy

```python
# Adjacency matrix
A = nx.to_numpy_array(G)                  # dense ndarray
G = nx.from_numpy_array(A)

# With specific node order
A = nx.to_numpy_array(G, nodelist=["A", "B", "C"])

# Weighted
A = nx.to_numpy_array(G, weight="weight")
```

### To/from scipy sparse

```python
from scipy.sparse import csr_array

sparse = nx.to_scipy_sparse_array(G)       # CSR format
G = nx.from_scipy_sparse_array(sparse)
```

---

## 13. Performance & Scaling

### NetworkX limits

NetworkX is **pure Python with dict-of-dicts storage**. It's optimized for flexibility, not raw speed.

| Graph size | NetworkX performance |
|------------|---------------------|
| < 10K nodes | Fast — no issues |
| 10K–100K nodes | Acceptable for most algorithms |
| 100K–1M nodes | Slow for complex algorithms (betweenness, all-pairs shortest path) |
| > 1M nodes | Consider backends or alternative libraries |

### Tips for better performance

```python
# 1. Use generators instead of lists when possible
for edge in nx.bfs_edges(G, "A"):     # lazy iteration
    ...

# 2. Use approximate algorithms for large graphs
nx.betweenness_centrality(G, k=500)   # sample 500 nodes instead of all

# 3. Convert to numpy/scipy for matrix operations
A = nx.to_scipy_sparse_array(G)

# 4. Use subgraphs to work on smaller portions
component = max(nx.connected_components(G), key=len)
sub = G.subgraph(component)
```

### Backends for acceleration (NetworkX 3.2+)

NetworkX supports backend dispatching — **same API, different engine**:

```bash
# GPU acceleration (requires NVIDIA GPU)
pip install nx-cugraph-cu12

# Parallel CPU processing
pip install nx-parallel

# GraphBLAS-based (linear algebra acceleration)
pip install graphblas-algorithms
```

```python
# Option 1: Environment variable (zero code change)
# NX_CUGRAPH_AUTOCONFIG=True python my_script.py

# Option 2: Explicit backend kwarg
nx.betweenness_centrality(G, backend="cugraph")
nx.pagerank(G, backend="cugraph")

# Option 3: Backend-specific graph type (auto-dispatch)
import nx_cugraph
nxcg_G = nx_cugraph.from_networkx(G)
nx.pagerank(nxcg_G)    # automatically uses cugraph
```

### Alternative libraries for very large graphs

| Library | Best for | Performance vs NetworkX |
|---------|----------|------------------------|
| **nx-cugraph** | GPU acceleration, same API | 50–500× faster |
| **graph-tool** | C++ core, statistical analysis | 10–100× faster |
| **igraph** | C core, general graph analysis | 10–50× faster |
| **SNAP** | Very large networks (100M+ edges) | Much faster |
| **DGL / PyG** | Graph neural networks | GPU-native |

---

## 14. Common Recipes

### Find shortest path and visualize it

```python
path = nx.shortest_path(G, "A", "D", weight="weight")
path_edges = list(zip(path, path[1:]))

pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color="lightgray")
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=3)
plt.show()
```

### Find most important nodes

```python
# Top 5 by betweenness centrality
bc = nx.betweenness_centrality(G)
top5 = sorted(bc, key=bc.get, reverse=True)[:5]
```

### Build a graph from a CSV

```python
import pandas as pd

df = pd.read_csv("edges.csv")  # columns: source, target, weight
G = nx.from_pandas_edgelist(df, "source", "target", edge_attr=["weight"])
```

### Detect communities and color them

```python
from networkx.algorithms.community import louvain_communities

communities = louvain_communities(G, seed=42)
color_map = {}
for i, comm in enumerate(communities):
    for node in comm:
        color_map[node] = i

colors = [color_map[n] for n in G.nodes()]
nx.draw(G, node_color=colors, cmap=plt.cm.tab10, with_labels=True)
plt.show()
```

### Build a dependency graph and get execution order

```python
DG = nx.DiGraph()
DG.add_edges_from([
    ("download_data", "clean_data"),
    ("clean_data", "train_model"),
    ("clean_data", "generate_report"),
    ("train_model", "evaluate"),
    ("generate_report", "evaluate"),
])

assert nx.is_directed_acyclic_graph(DG)

execution_order = list(nx.topological_sort(DG))
# ['download_data', 'clean_data', 'train_model', 'generate_report', 'evaluate']

# Parallel-safe execution order (grouped by generation)
for generation in nx.topological_generations(DG):
    print("Can run in parallel:", generation)
```

### Filter graph by edge weight

```python
# Keep only edges with weight > threshold
threshold = 0.5
edges_to_keep = [(u, v) for u, v, d in G.edges(data=True)
                  if d.get("weight", 0) > threshold]
filtered = G.edge_subgraph(edges_to_keep).copy()
```

### Save and restore graph layouts

```python
# Compute layout once, save on graph
pos = nx.spring_layout(G, seed=42)
for node, (x, y) in pos.items():
    G.nodes[node]["x"] = x
    G.nodes[node]["y"] = y

# Save to file
nx.write_graphml(G, "graph_with_layout.graphml")

# Restore layout
G2 = nx.read_graphml("graph_with_layout.graphml")
pos = {n: (G2.nodes[n]["x"], G2.nodes[n]["y"]) for n in G2.nodes()}
nx.draw(G2, pos)
```

---

## 15. Common Pitfalls

### Views vs copies

```python
# ⚠️ Subgraphs are VIEWS — they reflect changes to the original
sub = G.subgraph([1, 2, 3])
G.nodes[1]["color"] = "red"
print(sub.nodes[1]["color"])    # "red" — sub sees the change!

# ✅ Use .copy() for independent subgraphs
sub = G.subgraph([1, 2, 3]).copy()
```

### MultiGraph algorithms

```python
# ⚠️ Many algorithms don't support MultiGraph
# nx.betweenness_centrality(MG)  → may fail or give unexpected results

# ✅ Convert to simple graph first
simple = nx.Graph(MG)    # collapses parallel edges
```

### Directed vs undirected

```python
# ⚠️ Don't use undirected algorithms on directed graphs without thinking
# nx.connected_components(DG)  → ERROR

# ✅ Use the directed versions
nx.weakly_connected_components(DG)
nx.strongly_connected_components(DG)

# Or convert explicitly
UG = DG.to_undirected()
nx.connected_components(UG)
```

### Floating point in weighted graphs

```python
# ⚠️ Floating point comparison can change shortest paths
eps = 1e-17
G.add_edge("A", "D", weight=0.3 + eps)  # might not equal path A→B→C→D = 0.3

# ✅ Round weights or use integer weights (multiply by 1000, etc.)
```

### The `weight` attribute is special

```python
# ⚠️ Algorithms use weight="weight" by default
G.add_edge("A", "B", weight=10)
G.add_edge("A", "C", weight=1)
G.add_edge("C", "B", weight=1)

nx.shortest_path(G, "A", "B")    # ["A", "C", "B"] — uses weights!
nx.shortest_path(G, "A", "B", weight=None)  # ["A", "B"] — ignores weights
```

### Node types matter

```python
# ⚠️ Nodes are compared by equality AND type
G.add_node(1)        # int
G.add_node("1")      # string — this is a DIFFERENT node!
print(G.number_of_nodes())  # 2
```

### Don't modify during iteration

```python
# ❌ Crashes or unexpected behavior
for node in G.nodes():
    if G.degree(node) == 0:
        G.remove_node(node)

# ✅ Collect first, then modify
to_remove = [n for n in G.nodes() if G.degree(n) == 0]
G.remove_nodes_from(to_remove)
```

---

## Algorithm → NetworkX Function Cheat Sheet

| Algorithm | NetworkX function |
|-----------|-------------------|
| BFS | `nx.bfs_edges()`, `nx.bfs_tree()` |
| DFS | `nx.dfs_edges()`, `nx.dfs_tree()` |
| Dijkstra | `nx.dijkstra_path()`, `nx.shortest_path(weight=...)` |
| Bellman-Ford | `nx.bellman_ford_path()` |
| Floyd-Warshall | `nx.floyd_warshall()` |
| A* | `nx.astar_path()` |
| Kruskal's MST | `nx.minimum_spanning_tree()` (default) |
| Prim's MST | `nx.minimum_spanning_tree(algorithm="prim")` |
| Topological sort | `nx.topological_sort()` |
| Kahn's algorithm | `nx.topological_sort()` (uses Kahn's internally) |
| PageRank | `nx.pagerank()` |
| Betweenness | `nx.betweenness_centrality()` |
| Louvain | `nx.community.louvain_communities()` |
| Max flow (Ford-Fulkerson) | `nx.maximum_flow()` |
| Max matching | `nx.max_weight_matching()` |
| Graph coloring | `nx.greedy_color()` |
| Tarjan's SCC | `nx.strongly_connected_components()` |
| Kosaraju's SCC | `nx.kosaraju_strongly_connected_components()` |
| Union-Find | Internal to MST algorithms |
| Eulerian circuit | `nx.eulerian_circuit()` |
| Hamiltonian path | Not in NetworkX (NP-hard) |

---

> **Rule of thumb:** Start with `nx.Graph()` or `nx.DiGraph()`. Use generators for testing. Use `weight` attribute for weighted graphs. Convert to pandas/numpy when you need matrix operations. Add a backend when performance matters.
