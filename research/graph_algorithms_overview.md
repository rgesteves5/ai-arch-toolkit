# Graph Algorithms Overview

Graph algorithms are fundamental in computer science and are used to solve problems involving networks, such as social connections, transportation routes, or web pages. Graphs consist of vertices (nodes) and edges (connections), and algorithms operate on directed or undirected, weighted or unweighted graphs. Below, I'll categorize the most common graph algorithms with brief descriptions and examples. This isn't an exhaustive list, but it covers the major ones.

## 1. Traversal and Search Algorithms
These explore graphs to visit nodes or find paths.

Breadth-First Search (BFS): Explores level by level; used for shortest paths in unweighted graphs.
Depth-First Search (DFS): Explores as far as possible along each branch; used for topological sorting or detecting cycles.
Bidirectional Search: Searches from both start and end nodes; efficient for finding shortest paths.

## 2. Shortest Path Algorithms
These find the minimal cost path between nodes in weighted graphs.

Dijkstra's Algorithm: Finds shortest paths from a source using a priority queue; assumes non-negative weights.
Bellman-Ford Algorithm: Handles negative weights and detects negative cycles; slower than Dijkstra's.
Floyd-Warshall Algorithm: Computes all-pairs shortest paths; works for dense graphs with possible negative weights.
A Search*: Heuristic-based extension of Dijkstra's; used in pathfinding (e.g., games or GPS).
Johnson's Algorithm: Efficient for all-pairs shortest paths in sparse graphs with possible negative weights.

## 3. Minimum Spanning Tree (MST) Algorithms
These find a subset of edges connecting all nodes with minimal total weight (no cycles).

Kruskal's Algorithm: Sorts edges by weight and adds them if no cycle forms; uses union-find data structure.
Prim's Algorithm: Grows the MST from a starting node using a priority queue.
Borůvka's Algorithm: Builds MST by repeatedly connecting components; efficient for parallel computing.

## 4. Network Flow Algorithms
These model flow through a network (e.g., capacity-constrained edges).

Ford-Fulkerson Method: Computes maximum flow using augmenting paths; basis for many flow algorithms.
Edmonds-Karp Algorithm: BFS-based implementation of Ford-Fulkerson; finds max flow in polynomial time.
Dinic's Algorithm: Level-graph based; faster for dense graphs.
Push-Relabel Algorithm: Uses preflow and relabeling; efficient for large graphs.

## 5. Connectivity and Component Algorithms
These identify connected parts or bridges in graphs.

Tarjan's Algorithm: Finds strongly connected components (SCCs) or articulation points in linear time.
Kosaraju's Algorithm: Two-pass DFS to find SCCs in directed graphs.
Hopcroft-Tarjan Algorithm: Planarity testing for graphs.
Union-Find (Disjoint Set Union - DSU): Tracks connected components; used in MST algorithms.

## 6. Topological and Ordering Algorithms
These order nodes based on dependencies (for directed acyclic graphs - DAGs).

Topological Sort: Linear ordering where for every edge u→v, u comes before v; via DFS or Kahn's algorithm (BFS-based).
Kahn's Algorithm: BFS-based topological sort using indegrees.

## 7. Graph Coloring and Partitioning Algorithms
These assign colors or partitions to nodes/edges.

Greedy Coloring: Assigns colors to nodes without adjacent same-color conflicts; approximates chromatic number.
Welsh-Powell Algorithm: Variant of greedy coloring sorted by degree.
Kernighan-Lin Algorithm: Graph partitioning to minimize cut size.

## 8. Matching and Covering Algorithms
These find pairings or coverings in bipartite or general graphs.

Hopcroft-Karp Algorithm: Maximum matching in bipartite graphs.
Blossom Algorithm (Edmonds' Matching): Maximum matching in general graphs.
Hungarian Algorithm: Assignment problem (bipartite matching with costs).

## 9. Centrality and Ranking Algorithms
These measure node importance.

PageRank: Ranks nodes based on link structure (e.g., Google's original algorithm).
Betweenness Centrality: Measures nodes on shortest paths between others.
Closeness Centrality: Based on average shortest path to other nodes.
Eigenvector Centrality: Importance based on connections to important nodes.

## 10. Other Specialized Algorithms

Eulerian Path/Circuit: Finds paths visiting every edge exactly once (e.g., Fleury's or Hierholzer's).
Hamiltonian Path/Circuit: Visits every node exactly once (NP-hard; no efficient general algorithm).
Chinese Postman Problem: Shortest closed walk covering every edge (for undirected graphs).
Traveling Salesman Problem (TSP) Approximations: Like Christofides' for metric TSP.
Community Detection: Louvain algorithm or Girvan-Newman for clustering nodes.
