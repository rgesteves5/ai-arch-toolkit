# Algorithms for Knowledge Representation: Complete Reference

## Overview

This document catalogs every major algorithm that operates on the data structures used to model cognitive processes, organized by the cognitive system each one parallels. Each section covers the algorithms, their computational properties, and their relationship to how the brain processes, stores, and retrieves knowledge.

---

## 1. Graph Algorithms (Semantic Networks)

### 1.1 Traversal Algorithms

#### Breadth-First Search (BFS)

- **What it does:** Explores all neighbors of a node before moving deeper, radiating outward level by level
- **Cognitive parallel:** Closest algorithmic parallel to **spreading activation** — activation radiates outward from a concept, reaching closely related concepts first
- **Implementation:** Uses a queue; marks nodes as visited to avoid cycles
- **Time complexity:** O(V + E) where V = vertices, E = edges
- **Space complexity:** O(V)
- **Key properties:** Finds shortest path in unweighted graphs; guarantees that closer nodes are visited first

#### Depth-First Search (DFS)

- **What it does:** Follows one path as deep as possible before backtracking
- **Cognitive parallel:** Resembles **chain-of-thought reasoning** — pursuing a single line of association to its conclusion before trying another
- **Implementation:** Uses a stack (or recursion); marks nodes as visited
- **Time complexity:** O(V + E)
- **Space complexity:** O(V) worst case; O(h) for recursion stack where h = max depth
- **Key properties:** Useful for topological sorting, cycle detection, and connected component identification

#### A* Search

- **What it does:** Finds the shortest path between two nodes using a heuristic estimate of remaining distance
- **Cognitive parallel:** How the brain uses prior knowledge (the heuristic) to guide search toward likely relevant memories, rather than searching exhaustively
- **Implementation:** Priority queue ordered by f(n) = g(n) + h(n), where g(n) is cost so far and h(n) is heuristic estimate
- **Time complexity:** Depends on heuristic quality; optimal with admissible heuristic
- **Key properties:** Generalizes Dijkstra's algorithm by adding heuristic guidance; optimal and complete with admissible heuristic

#### Dijkstra's Algorithm

- **What it does:** Finds shortest path from a source node to all other nodes in a weighted graph (non-negative weights)
- **Cognitive parallel:** In a semantic network with weighted associations, finds the "closest" concept to a starting concept where closeness = cumulative association strength
- **Implementation:** Priority queue (min-heap); greedily selects the unvisited node with smallest known distance
- **Time complexity:** O((V + E) log V) with binary heap
- **Key properties:** Guarantees optimal shortest paths for non-negative edge weights

### 1.2 Shortest Path Algorithms

#### Bellman-Ford Algorithm

- **What it does:** Finds shortest paths from a single source, handling negative edge weights and detecting negative cycles
- **Cognitive parallel:** More general path-finding that can handle "negative associations" — situations where traversing certain links reduces overall cognitive distance
- **Time complexity:** O(V × E)
- **Key properties:** Slower than Dijkstra but handles negative weights; detects negative cycles

#### Floyd-Warshall Algorithm

- **What it does:** Computes shortest paths between **all pairs** of nodes
- **Cognitive parallel:** Precomputing the "semantic distance" between every pair of concepts — a complete semantic similarity matrix
- **Time complexity:** O(V³)
- **Space complexity:** O(V²)
- **Key properties:** Simple to implement; dynamic programming approach; works with negative edges (no negative cycles)

### 1.3 Centrality and Importance Algorithms

#### PageRank

- **What it does:** Computes the importance of nodes based on the link structure — a node is important if other important nodes link to it
- **Cognitive parallel:** Identifying the most central, densely connected concepts in a knowledge network — hubs like "animal," "cause," or "energy" that are activated across many contexts
- **Implementation:** Iterative computation: PR(i) = (1-d)/N + d × Σ(PR(j)/L(j)) for all j linking to i; d = damping factor (typically 0.85)
- **Time complexity:** O(V + E) per iteration; typically converges in ~50 iterations
- **Key properties:** Originally Google's ranking algorithm; models random walk on a graph

#### Betweenness Centrality

- **What it does:** Identifies nodes that serve as bridges between different clusters — nodes that appear on many shortest paths between other nodes
- **Cognitive parallel:** Concepts that connect otherwise separate knowledge domains — like "energy" bridging physics and biology, or "structure" bridging architecture and programming
- **Time complexity:** O(V × E) using Brandes' algorithm
- **Key properties:** High betweenness = critical bridge concept; removing it would fragment the network

### 1.4 Community Detection Algorithms

#### Louvain Algorithm

- **What it does:** Finds communities (clusters of densely connected nodes) by maximizing modularity
- **Cognitive parallel:** Discovering knowledge domains — groups of concepts that are heavily interconnected internally but loosely connected to other groups (like "biology concepts" vs "music concepts")
- **Implementation:** Iteratively merges nodes into communities, then treats communities as nodes and repeats
- **Time complexity:** O(n log n) in practice
- **Key properties:** Fast, scalable; widely used for large networks

#### Girvan-Newman Algorithm

- **What it does:** Identifies communities by progressively removing edges with highest betweenness centrality
- **Cognitive parallel:** Discovering the natural fault lines between knowledge domains — the connections that, if severed, would most cleanly separate conceptual clusters
- **Time complexity:** O(V × E²)
- **Key properties:** Produces a hierarchical decomposition (dendrogram) of the network

### 1.5 Graph Construction Algorithms

#### Kruskal's Algorithm

- **What it does:** Finds the minimum spanning tree — the subset of edges that connects all nodes with minimum total weight
- **Cognitive parallel:** The most efficient way to connect all concepts in a curriculum — the minimal set of associations needed to link everything together
- **Implementation:** Sort edges by weight; add edges greedily if they don't create a cycle (using Union-Find)
- **Time complexity:** O(E log E)

#### Prim's Algorithm

- **What it does:** Also finds minimum spanning tree, growing from a starting node
- **Cognitive parallel:** Building a knowledge network outward from a starting concept, always adding the nearest unconnected concept
- **Implementation:** Priority queue of edges from the current tree to unvisited nodes
- **Time complexity:** O((V + E) log V) with binary heap

---

## 2. Tree Algorithms (Taxonomic / Hierarchical Knowledge)

### 2.1 Search Algorithms

#### Binary Search (on BSTs)

- **What it does:** Finds a target value by repeatedly comparing with the middle element and eliminating half the search space
- **Cognitive parallel:** Hierarchical categorization — narrowing down from broad categories to specific instances through a series of binary decisions
- **Time complexity:** O(log n) for balanced trees; O(n) worst case for unbalanced
- **Key properties:** Requires ordered data; extremely efficient for sorted collections

#### AVL Tree Rotations

- **What it does:** Self-balancing algorithm that maintains the height difference between left and right subtrees at ≤ 1
- **Cognitive parallel:** How the brain reorganizes category boundaries to keep classification efficient — if one category becomes too large, subcategories emerge to maintain balance
- **Operations:** Single rotation (left/right) and double rotation (left-right/right-left)
- **Time complexity:** O(log n) guaranteed for search, insert, delete

#### Red-Black Tree Operations

- **What it does:** Self-balancing BST with color constraints (each node is red or black) ensuring the tree stays approximately balanced
- **Cognitive parallel:** A more relaxed balancing strategy — allows some imbalance for faster insertion/deletion (less reorganization per operation)
- **Time complexity:** O(log n) guaranteed for all operations
- **Key properties:** Used extensively in standard library implementations (Java TreeMap, C++ std::map)

#### B-Tree Search and Insertion

- **What it does:** Multi-way search tree where each node can have many children, keeping the tree very shallow
- **Cognitive parallel:** Expert knowledge organized in broad, shallow hierarchies — an expert can quickly narrow to the right area because their categories are efficiently organized
- **Time complexity:** O(log n) with very small constants due to low tree height
- **Key properties:** Optimized for disk access; foundation of most database index structures

### 2.2 Traversal Algorithms

#### Pre-order Traversal (Root → Left → Right)

- **What it does:** Visits the root first, then recursively visits children
- **Cognitive parallel:** **Top-down reasoning** — starting from a general category and drilling down to specifics
- **Use cases:** Creating a copy of the tree; serialization; prefix expression evaluation

#### In-order Traversal (Left → Root → Right)

- **What it does:** Visits left subtree, then root, then right subtree
- **Cognitive parallel:** Produces sorted output from a BST — like mentally listing items in order
- **Use cases:** Sorted enumeration; verifying BST property

#### Post-order Traversal (Left → Right → Root)

- **What it does:** Visits children before the root
- **Cognitive parallel:** **Bottom-up induction** — gathering specific examples before forming a general concept
- **Use cases:** Deletion; computing dependent values (child values needed before parent)

#### Level-order Traversal (BFS on trees)

- **What it does:** Visits all nodes at depth d before any nodes at depth d+1
- **Cognitive parallel:** Processing all concepts at the same level of abstraction before going deeper
- **Implementation:** Queue-based BFS

### 2.3 Trie Operations

#### Trie Insertion and Search

- **What it does:** Stores and retrieves strings character by character along a path from root to leaf
- **Cognitive parallel:** **Cohort activation** in speech recognition — hearing "pre-" activates "predict," "prepare," "present," "pretty" simultaneously
- **Time complexity:** O(m) where m = key length
- **Key properties:** Prefix matching is natural and efficient; no hash collisions

#### Prefix Matching on Tries

- **What it does:** Finds all entries sharing a common prefix
- **Cognitive parallel:** Hearing the beginning of a word activates all candidate completions
- **Time complexity:** O(p + k) where p = prefix length, k = number of matches
- **Use cases:** Autocomplete, spell checking, IP routing

#### Radix Tree (Patricia Trie) Compression

- **What it does:** Merges nodes with single children, creating a more compact representation
- **Cognitive parallel:** Frequently used knowledge pathways become compressed and automatized — you don't traverse every intermediate step
- **Key properties:** More space-efficient than standard tries; same time complexity

### 2.4 Tree Construction Algorithms

#### Huffman Tree Construction

- **What it does:** Builds an optimal prefix-free binary code based on symbol frequency — more frequent symbols get shorter codes
- **Cognitive parallel:** How the brain allocates representational resources — more commonly encountered concepts have faster, more efficient access paths. Linguistically manifests as **Zipf's Law** — more frequent words tend to be shorter ("the," "a," "is" vs. "phenomenological," "discombobulate")
- **Implementation:** Priority queue; repeatedly merge the two least frequent nodes
- **Time complexity:** O(n log n)
- **Key properties:** Produces provably optimal prefix-free codes; foundation of many compression algorithms

---

## 3. Hash Map Algorithms (Associative Memory)

### 3.1 Hashing Functions

**Purpose:** Convert arbitrary keys into fixed-size indices for fast lookup

**Common hash functions:** MD5, SHA-family, MurmurHash, CityHash, xxHash, FNV

**Cognitive parallel:** How a partial cue (a smell, a few musical notes) gets transformed into an access point for a complete memory. A good hash function distributes keys uniformly — effective memory cues point to distinct memories without excessive overlap.

**Properties of good hash functions:**
- Deterministic (same input always produces same output)
- Uniform distribution (minimizes collisions)
- Avalanche effect (small changes in input produce large changes in output)
- Fast to compute

### 3.2 Collision Resolution

#### Separate Chaining

- **What it does:** Stores multiple entries at the same hash index using a linked list (or tree for large chains)
- **Cognitive parallel:** A single cue activating multiple competing memories — you hear a name and several people with that name come to mind; you traverse the "chain" to find the right one
- **Time complexity:** O(1) average; O(n/m) with load factor n/m

#### Open Addressing — Linear Probing

- **What it does:** When a collision occurs, searches sequentially for the next empty slot
- **Cognitive parallel:** When your first retrieval attempt hits the wrong memory, you systematically search nearby associations
- **Issue:** Primary clustering — runs of occupied slots form, degrading performance

#### Open Addressing — Quadratic Probing

- **What it does:** Probes at positions h+1², h+2², h+3², etc.
- **Cognitive parallel:** Searching with increasing steps away from the initial association
- **Advantage:** Reduces primary clustering

#### Open Addressing — Double Hashing

- **What it does:** Uses a second hash function to determine the probe interval
- **Cognitive parallel:** Using a different retrieval strategy when the first fails
- **Advantage:** Minimizes clustering

#### Cuckoo Hashing

- **What it does:** Uses two hash functions and two tables; displaces existing entries to make room for new ones
- **Cognitive parallel:** **Memory interference** — new memories can displace or disrupt old ones that occupy similar representational space
- **Time complexity:** O(1) worst-case lookup; amortized O(1) insertion
- **Key properties:** Guarantees constant-time lookup

### 3.3 Consistent Hashing

- **What it does:** Distributes data across a dynamic set of nodes on a hash ring, minimizing redistribution when nodes are added/removed
- **Cognitive parallel:** How the brain distributes knowledge across neural populations and gracefully handles the loss of some neurons without catastrophic failure
- **Key properties:** Only K/n keys need to be remapped when a node is added (K = total keys, n = total nodes)

### 3.4 Bloom Filter Algorithms

#### Bloom Filter Insertion and Membership Testing

- **What it does:** Sets bits at positions determined by k hash functions to insert; tests same positions for membership
- **Cognitive parallel:** **Recognition memory** — fast, fuzzy familiarity testing; false positives possible (thinking something is familiar when it isn't) but no false negatives (if it's truly novel, you know)
- **Time complexity:** O(k) for both insertion and query (k = number of hash functions)
- **Space efficiency:** Far less memory than storing actual elements
- **False positive rate:** (1 − e^(−kn/m))^k where n = elements, m = bits, k = hash functions

#### Counting Bloom Filters

- **What it does:** Extends Bloom filters by using counters instead of single bits, allowing deletions
- **Cognitive parallel:** Familiarity signals that can strengthen or weaken over time
- **Trade-off:** More space (counters instead of bits) but supports deletion

---

## 4. Stack, Queue, and Buffer Algorithms (Working Memory)

### 4.1 Stack Algorithms

#### Push and Pop

- **What it does:** O(1) operations to add/remove from top of stack
- **Cognitive parallel:** Items entering and leaving the focus of working memory

#### Call Stack Management

- **What it does:** Tracks function calls and returns; each invocation pushes a frame, each return pops it
- **Cognitive parallel:** Nested cognitive processing — thinking about X which requires thinking about Y which requires thinking about Z
- **Stack overflow:** Exceeding the stack depth limit — cognitive overload from too many nested tasks

#### Shunting-Yard Algorithm (Dijkstra)

- **What it does:** Converts infix mathematical expressions to postfix notation using a stack, respecting operator precedence
- **Cognitive parallel:** How the brain parses and evaluates complex nested expressions in language and mathematics, handling precedence and grouping
- **Time complexity:** O(n)
- **Key properties:** Eliminates the need for parentheses; produces unambiguous evaluation order

#### Backtracking Algorithms

- **What it does:** Explores possibilities using the stack (recursion), undoing choices that don't work and trying alternatives
- **Cognitive parallel:** Trial-and-error reasoning in working memory — mentally trying an approach, hitting a dead end, and backing up
- **Examples:** N-Queens problem, Sudoku solving, maze solving, constraint satisfaction
- **Implementation:** Recursive DFS with constraint checking at each step

### 4.2 Queue and Priority Queue Algorithms

#### Enqueue and Dequeue

- **What it does:** O(1) operations for FIFO processing
- **Cognitive parallel:** Sensory input processed roughly in order of arrival

#### Binary Heap Operations (for Priority Queues)

**Heap Insertion (Sift-Up / Bubble-Up):**
- Add element at bottom of heap; swap with parent while it violates heap property
- **Time complexity:** O(log n)
- **Cognitive parallel:** A new, high-priority item entering working memory and rising to the top of attentional focus

**Extract-Min/Max (Sift-Down):**
- Remove root (highest-priority element); replace with last element; sift down
- **Time complexity:** O(log n)
- **Cognitive parallel:** Processing and removing the most important item from the attention queue, then reorganizing priorities

**Heapify (Build Heap):**
- Convert an arbitrary array into a valid heap
- **Time complexity:** O(n) — surprisingly linear
- **Cognitive parallel:** Rapidly organizing a set of competing demands by priority

### 4.3 Circular Buffer Algorithms

#### Overwrite-on-Full Policy

- **What it does:** When buffer is full, newest item overwrites the oldest
- **Cognitive parallel:** Almost exact model of displacement in short-term memory — new information pushes out old
- **Implementation:** Two pointers (read/write) on a fixed-size array, wrapping around
- **Time complexity:** O(1) for read and write
- **Fixed size:** Corresponds to working memory capacity limit (~4 items)

---

## 5. Linked List Algorithms (Episodic Memory Chains)

#### Insertion and Deletion

- **What it does:** O(1) at known positions; O(n) to find the position
- **Cognitive parallel:** New experiences seamlessly added to the temporal stream of episodic memory

#### Forward and Backward Traversal (Doubly Linked Lists)

- **What it does:** Follow next/previous pointers to traverse in either direction
- **Cognitive parallel:** **Mental time travel** — replaying experiences forward or tracing back to predecessors

#### Skip Lists

- **What it does:** Adds express lanes over a linked list with multiple levels of forward pointers, allowing O(log n) expected search
- **Cognitive parallel:** Skipping over uneventful stretches of time to land on significant memories — you don't replay every moment of the past year to find your birthday
- **Implementation:** Probabilistic data structure; each node has a random number of forward pointers
- **Time complexity:** O(log n) expected search, insert, delete
- **Key properties:** Simpler to implement than balanced BSTs; good cache performance

---

## 6. Matrix and Tensor Algorithms (Distributed / Neural Representation)

### 6.1 Matrix Operations

#### Matrix Multiplication

- **What it does:** The fundamental operation in neural networks — transforming input vectors through learned weight matrices
- **Cognitive parallel:** Every forward pass of a neural network (and by analogy, every act of perception or cognition) involves series of transformations of activation patterns
- **Time complexity:** O(n³) naive; O(n^2.81) Strassen's; O(n^2.37) Coppersmith-Winograd family (theoretical)
- **Strassen's Algorithm:** Reduces the number of multiplications needed by cleverly decomposing matrices

#### Singular Value Decomposition (SVD)

- **What it does:** Decomposes a matrix A into UΣVᵀ — revealing the fundamental components of high-dimensional data
- **Cognitive parallel:** Extracting the latent dimensions along which concepts vary — the underlying structure of meaning
- **Application in cognitive science:** **Latent Semantic Analysis (LSA)** uses SVD on a term-document matrix to discover semantic relationships between words without any explicit knowledge of grammar or meaning
- **Time complexity:** O(min(mn², m²n)) for an m×n matrix

#### Principal Component Analysis (PCA)

- **What it does:** Finds the directions of maximum variance in data, projecting it onto a lower-dimensional space
- **Cognitive parallel:** How the brain might extract the most informative features from experience and compress representations to their most important dimensions
- **Implementation:** Compute covariance matrix, then find its eigenvalues/eigenvectors (or use SVD directly)
- **Key properties:** Dimensionality reduction; finds the axes of greatest variability

#### Eigenvalue Decomposition

- **What it does:** Decomposes a square matrix into its eigenvalues and eigenvectors — the fundamental modes of the system
- **Cognitive parallel:** In network neuroscience, eigenvalues of brain connectivity matrices reveal the dominant patterns of correlated activity
- **Key properties:** Eigenvalues indicate the importance of each mode; eigenvectors indicate the pattern

### 6.2 Tensor Operations

#### Tensor Decomposition (CP, Tucker)

- **What it does:** Extends matrix factorization to higher dimensions, decomposing multi-dimensional data into component factors
- **Cognitive parallel:** Knowledge is multi-dimensional — a concept has visual features, acoustic features, motor associations, emotional valence, and categorical memberships simultaneously; tensor methods capture these multi-way relationships
- **CP decomposition:** Expresses a tensor as a sum of rank-one tensors
- **Tucker decomposition:** Generalizes PCA to higher dimensions

#### Einstein Summation (einsum)

- **What it does:** Generalized notation for tensor contractions that efficiently expresses many neural network operations
- **Key properties:** Subsumes matrix multiplication, trace, transpose, outer product, and many other operations in a single notation

---

## 7. Neural Network Algorithms (Connectionism)

### 7.1 Training Algorithms

#### Backpropagation

- **What it does:** Computes the gradient of the loss function with respect to every weight in the network by applying the chain rule of calculus backward through the layers
- **Cognitive parallel:** Debated — the exact algorithm seems biologically implausible, but algorithms like **feedback alignment** and **predictive coding** may serve similar functions in the brain
- **Implementation:** Forward pass computes outputs; backward pass computes gradients layer by layer from output to input
- **Time complexity:** O(n) where n = number of weights (one forward + one backward pass)
- **Key insight:** Each weight learns how much it contributed to the overall error and adjusts proportionally

#### Stochastic Gradient Descent (SGD) and Variants

**Basic SGD:**
- Updates weights in the direction that reduces error: w ← w − η∇L(w)
- Uses a random subset (mini-batch) of data for each update
- Noise from mini-batch sampling can help escape local minima

**SGD with Momentum:**
- Accumulates velocity from past gradients: v ← βv + η∇L(w); w ← w − v
- Helps push through shallow local minima and accelerates convergence
- **Cognitive parallel:** Repeated exposure to consistent patterns builds learning "momentum"

**Adam (Adaptive Moment Estimation):**
- Adapts the learning rate for each weight individually based on first and second moments of gradients
- **Cognitive parallel:** How the brain might adjust learning rates differently for different synapses based on their activation history
- **Key properties:** Combines momentum with per-parameter learning rates; widely used default optimizer

**Learning Rate Scheduling:**
- Starts with large learning rates and gradually reduces them
- Strategies: step decay, exponential decay, cosine annealing, warm-up then decay
- **Cognitive parallel:** **Critical periods** in brain development — the brain is initially highly plastic and gradually becomes more stable

### 7.2 Regularization Algorithms

#### Dropout

- **What it does:** Randomly deactivates neurons during training (each with probability p), forcing the network to develop redundant representations
- **Cognitive parallel:** How the brain maintains robustness — knowledge distributed across many neurons so loss of some doesn't destroy entire memories
- **Key properties:** Acts as ensemble training; reduces overfitting; typically p = 0.5 for hidden layers

#### Weight Decay (L2 Regularization)

- **What it does:** Adds a penalty term proportional to the sum of squared weights: L_total = L_data + λΣw²
- Pushes the network toward smaller weights and simpler solutions
- **Cognitive parallel:** **Synaptic scaling** — a homeostatic mechanism that prevents runaway excitation by proportionally scaling down all synaptic weights

#### Batch Normalization

- **What it does:** Normalizes activations within each layer to have zero mean and unit variance, then applies learned scale and shift
- **Cognitive parallel:** Homeostatic mechanisms in the brain that keep neural firing rates within functional ranges
- **Key properties:** Stabilizes training; allows higher learning rates; acts as mild regularization

### 7.3 Specific Architecture Algorithms

#### Convolution Operation (CNNs)

- **What it does:** Slides a small filter (kernel) across input data to detect local patterns; each filter detects a specific feature regardless of position
- **Cognitive parallel:** How the visual cortex processes information through **receptive fields** — each neuron responds to a small region; successive layers detect increasingly complex features (edges → textures → parts → objects)
- **Key properties:** Parameter sharing (same filter everywhere); translation invariance; hierarchical feature detection
- **Implementation:** Element-wise multiplication of filter with input patch, summed into a single output value; repeated at every position

#### Backpropagation Through Time — BPTT (RNNs)

- **What it does:** Unrolls the recurrent network through time and applies standard backpropagation to the unrolled graph
- **Cognitive parallel:** Learning from sequences of experience — updating your model of how events unfold over time
- **Challenge:** Vanishing/exploding gradients over long sequences — addressed by gating mechanisms

#### LSTM Gating Mechanism

- **What it does:** Uses three gates to selectively control information flow through a recurrent network:
  - **Forget gate:** Decides what information to discard from the cell state — f = σ(W_f · [h_{t-1}, x_t] + b_f)
  - **Input gate:** Decides what new information to store — i = σ(W_i · [h_{t-1}, x_t] + b_i)
  - **Output gate:** Decides what information to output — o = σ(W_o · [h_{t-1}, x_t] + b_o)
- **Cognitive parallel:** The forget gate directly models selective forgetting; the input gate models selective encoding; the output gate models selective retrieval
- **Key properties:** Maintains information over long sequences; solves the vanishing gradient problem

#### GRU (Gated Recurrent Unit)

- **What it does:** Simplified version of LSTM with two gates (reset and update) instead of three
- **Key properties:** Similar performance to LSTM with fewer parameters; faster to train

#### Self-Attention (Transformers)

- **What it does:** Computes how relevant each element in a sequence is to every other element:
  - Attention(Q, K, V) = softmax(QKᵀ / √d) × V
  - Q (queries), K (keys), V (values) are linear projections of the input
- **Cognitive parallel:** Flexible allocation of processing resources based on relevance — attending to the most informative parts of the input regardless of their position
- **Multi-head attention:** Runs several attention computations in parallel, each attending to different types of relationships — analogous to different brain systems processing different aspects of a stimulus simultaneously
- **Time complexity:** O(n²d) where n = sequence length, d = dimension
- **Key properties:** Captures long-range dependencies; parallelizable (unlike RNNs); foundation of modern large language models

---

## 8. Associative Memory Algorithms (Pattern Completion)

### 8.1 Hopfield Network

#### Energy Minimization and Pattern Retrieval

- **What it does:** Stores patterns as energy minima in a dynamical system; retrieval starts from a partial or noisy pattern and lets the network settle to the nearest stored pattern
- **Update rule:** sᵢ ← sign(Σⱼ wᵢⱼsⱼ − θᵢ) — each neuron updates based on the weighted sum of all other neurons
- **Cognitive parallel:** Directly models **pattern completion** in the hippocampus — providing a partial cue and recovering the complete memory
- **Storage capacity:** ~0.14N patterns for N neurons (classic result)
- **Key properties:** Guaranteed convergence (energy always decreases); content-addressable retrieval

#### Hebbian Learning (Weight Setting)

- **What it does:** Sets weights based on correlation between stored patterns: wᵢⱼ = (1/N) Σμ ξᵢμ ξⱼμ
- **Cognitive parallel:** Direct implementation of "neurons that fire together wire together"
- **Key properties:** Simplest and most biologically plausible learning rule; one-shot learning (no iterative training)

### 8.2 Modern Hopfield Networks

- **What it does:** Uses exponential interaction functions instead of quadratic, allowing exponentially many stored patterns
- **Key insight:** The update rule of modern Hopfield networks is **mathematically equivalent to the attention mechanism in transformers** — a deep connection between associative memory and modern AI
- **Storage capacity:** Exponential in N (vs. linear for classical Hopfield)

### 8.3 Sparse Distributed Memory (SDM)

- **What it does:** Stores and retrieves patterns in a high-dimensional binary space using a set of randomly positioned "hard locations"
- **Proposed by:** Pentti Kanerva
- **Cognitive parallel:** Explicitly designed to model human memory — shares properties of graceful degradation, content-addressability, and interference between similar patterns
- **Key properties:** Robust to noise; natural pattern completion; handles high-dimensional data efficiently

---

## 9. Finite State Machine and Decision Tree Algorithms (Procedural Knowledge)

### 9.1 FSM Algorithms

#### State Transition Evaluation

- **What it does:** Given current state and input, looks up next state in a transition table
- **Time complexity:** O(1)
- **Cognitive parallel:** Automatic, well-learned procedural responses — immediate, no deliberation

#### NFA to DFA Conversion (Subset Construction)

- **What it does:** Converts a nondeterministic finite automaton (multiple possible transitions) into a deterministic one (exactly one transition per input)
- **Cognitive parallel:** **Skill automatization** — early in learning, multiple possible actions are considered (nondeterministic); with practice, the correct action becomes automatic (deterministic)
- **Time complexity:** O(2^n) worst case for n NFA states; usually much smaller in practice

#### FSM Minimization (Hopcroft's Algorithm)

- **What it does:** Finds the smallest DFA that produces the same behavior as a given DFA
- **Cognitive parallel:** How skills become more efficient with practice — eliminating unnecessary decision points, streamlining procedures
- **Time complexity:** O(n log n) for n states

### 9.2 Decision Tree Algorithms

#### ID3 (Iterative Dichotomiser 3)

- **What it does:** Builds a decision tree by recursively selecting the feature with the highest **information gain** (entropy reduction) at each node
- **Cognitive parallel:** How experts learn to ask the most diagnostic questions first — questions that most efficiently narrow down possibilities
- **Information gain:** IG(S, A) = H(S) − Σ(|Sᵥ|/|S|) × H(Sᵥ) where H = entropy
- **Limitations:** Only handles categorical features; prone to overfitting

#### C4.5

- **What it does:** Improves on ID3 by handling continuous features, missing data, and using gain ratio instead of information gain
- **Key properties:** Uses pruning to reduce overfitting; handles real-world data better than ID3

#### CART (Classification and Regression Trees)

- **What it does:** Builds binary decision trees using **Gini impurity** as the splitting criterion
- **Gini impurity:** G = 1 − Σpᵢ² (probability of misclassification)
- **Key properties:** Handles both classification and regression; always binary splits

#### Random Forests

- **What it does:** Builds many decision trees on random subsets of features and data; aggregates predictions through majority voting (classification) or averaging (regression)
- **Cognitive parallel:** Decision-making by combining input from multiple specialized subsystems, each attending to different features — ensemble wisdom
- **Key properties:** Reduces overfitting; handles high-dimensional data; provides feature importance rankings

#### Gradient Boosted Trees (XGBoost, LightGBM, CatBoost)

- **What it does:** Builds trees sequentially, where each new tree corrects the errors (residuals) of the ensemble so far
- **Cognitive parallel:** Iterative skill refinement — each practice session focuses on correcting mistakes from the last one
- **Key properties:** Often achieves state-of-the-art performance on tabular data; highly tunable

#### Decision Tree Pruning

**Reduced-Error Pruning:**
- Remove branches that don't improve accuracy on a validation set
- **Cognitive parallel:** Simplifying over-detailed rules through experience

**Cost-Complexity Pruning (Minimal Cost-Complexity):**
- Balances tree complexity against accuracy using a complexity parameter α
- Produces a sequence of increasingly pruned trees; selects the best via cross-validation
- **Cognitive parallel:** Learning which distinctions matter and which can be ignored

---

## 10. Generative and Compression Algorithms (Predictive Coding)

### 10.1 Generative Algorithms

#### Variational Autoencoders (VAEs)

- **What it does:** Learns a compressed latent representation of data (encoder) and generates new samples from it (decoder); trained to maximize a variational lower bound (ELBO) on the data likelihood
- **Cognitive parallel:** The encoder models how the brain extracts abstract representations from experience; the decoder models how predictions are generated from internal representations
- **Training objective:** Explicitly balances reconstruction accuracy against compression — mirrors the brain's trade-off between detail and efficiency
- **Key properties:** Generates new samples; learns smooth, continuous latent spaces; probabilistic framework

#### Generative Adversarial Networks (GANs)

- **What it does:** Trains two networks adversarially — a generator creates fake data, a discriminator tries to distinguish real from fake
- **Cognitive parallel:** How the brain refines its internal models by comparing predictions against actual sensory input — the "discriminator" is the error-detection mechanism
- **Training:** Minimax game: min_G max_D E[log D(x)] + E[log(1 − D(G(z)))]
- **Key properties:** Produces highly realistic samples; training can be unstable; mode collapse risk

#### Diffusion Models

- **What it does:** Gradually adds noise to data in a forward process, then learns to reverse the noise (denoise) step by step
- **Cognitive parallel:** Memory retrieval as progressive refinement — starting from a vague, noisy activation and iteratively denoising into a clear memory
- **Key properties:** State-of-the-art image generation; theoretically principled; stable training

#### Boltzmann Machines and Restricted Boltzmann Machines (RBMs)

- **What it does:** Stochastic generative models that learn probability distributions over data through energy-based learning
- **Training:** **Contrastive divergence** (CD) — approximates the gradient of the log-likelihood
- **Cognitive parallel:** Early models designed to capture how the brain learns the statistical structure of the environment
- **RBMs:** Restricted version with no intra-layer connections; easier to train; used as building blocks for deep belief networks

### 10.2 Compression Algorithms

#### Lempel-Ziv Family (LZ77, LZ78, LZW)

- **What it does:** Compresses data by finding repeated patterns and replacing them with references to earlier occurrences
- **Cognitive parallel:** How the brain compresses experience by extracting regularities and storing deviations from patterns — you don't re-encode the familiar parts
- **Foundation of:** ZIP, GZIP, PNG compression
- **Key properties:** Universal (no prior knowledge of data distribution needed); dictionary-based

#### Arithmetic Coding

- **What it does:** Encodes entire messages as single numbers within a probability range, achieving near-optimal compression
- **Cognitive parallel:** The better the model predicts the next symbol, the more efficient the compression — directly paralleling how better predictive models in the brain enable more efficient representation
- **Key properties:** Approaches the theoretical entropy limit; uses a probabilistic model of the data

#### Delta Encoding

- **What it does:** Stores only the differences between successive data points
- **Cognitive parallel:** The algorithmic equivalent of **prediction error coding** — you don't represent the full sensory input, just the difference between prediction and reality
- **Use cases:** Version control, video compression (store frame differences), time-series data

#### Run-Length Encoding (RLE)

- **What it does:** Compresses sequences of repeated values by storing (value, count) pairs
- **Cognitive parallel:** How the brain efficiently represents stable, unchanging input — you don't reprocess the static background of your visual field every millisecond
- **Key properties:** Very simple; effective for data with long runs of repeated values

---

## 11. Caching and Indexing Algorithms (Memory Consolidation)

### 11.1 Cache Replacement Algorithms

#### LRU (Least Recently Used)

- **What it does:** Evicts the item that hasn't been accessed for the longest time
- **Cognitive parallel:** Memories not retrieved recently are more likely to be forgotten
- **Implementation:** Hash map + doubly linked list for O(1) access and eviction
- **Key properties:** Simple, effective; exploits temporal locality

#### LFU (Least Frequently Used)

- **What it does:** Evicts the item accessed least often overall
- **Cognitive parallel:** Rarely activated memories fade while frequently retrieved ones strengthen
- **Implementation:** Hash map + frequency counters + min-heap or frequency lists
- **Key properties:** Better for stable access patterns; slower to adapt to changes

#### ARC (Adaptive Replacement Cache)

- **What it does:** Dynamically balances between recency and frequency using two LRU lists and shadow entries
- **Cognitive parallel:** More closely models actual memory, which considers both how recently and how frequently something was encountered
- **Key properties:** Self-tuning; outperforms LRU and LFU in many workloads

### 11.2 Database Indexing Algorithms

#### B-Tree Indexing

- **What it does:** Organizes data for efficient range queries and ordered access using balanced multi-way trees
- **Cognitive parallel:** As knowledge consolidates, the brain builds richer retrieval cues and more organized pathways
- **Time complexity:** O(log n) for search, insert, delete

#### Hash Indexing

- **What it does:** Provides O(1) lookup for exact-match queries
- **Cognitive parallel:** Direct cue-based retrieval — a specific cue instantly activates a specific memory

#### Inverted Indexing

- **What it does:** Maps from content terms to the documents containing them (reversed from the usual document→terms mapping)
- **Cognitive parallel:** Having a mental index from features to concepts — seeing "stripes" activates "zebra," "tiger," "barber pole"
- **Used by:** Search engines (Google, Elasticsearch, Lucene)
- **Key properties:** Enables full-text search; foundation of information retrieval

---

## 12. Garbage Collection Algorithms (Forgetting and Pruning)

#### Mark-and-Sweep

- **What it does:** Starting from root references, traverses all reachable objects; everything not reached is freed
- **Cognitive parallel:** **Synaptic pruning** — connections not part of active circuits get eliminated, especially during sleep and adolescent development
- **Time complexity:** O(n) where n = total objects
- **Key properties:** Handles cycles; requires stopping the program (stop-the-world pauses)

#### Reference Counting

- **What it does:** Each object tracks how many references point to it; when count reaches zero, the object is freed immediately
- **Cognitive parallel:** Memories with zero remaining associations — no cues leading to them — become irretrievable
- **Time complexity:** O(1) for increment/decrement
- **Limitation:** Cannot handle circular references without cycle detection

#### Generational Garbage Collection

- **What it does:** Divides objects into generations (young, old, sometimes permanent); collects young generations more frequently
- **Cognitive parallel:** Recent, unconsolidated memories are more vulnerable to loss; old, well-consolidated memories are durable and rarely need attention
- **Key insight:** Most objects die young (weak generational hypothesis) — most memories are forgotten quickly
- **Implementation:** Young generation uses copying collection; old generation uses mark-and-sweep

#### Compaction

- **What it does:** After garbage collection, moves surviving objects together to eliminate memory fragmentation
- **Cognitive parallel:** Memory reconsolidation during sleep — memories are reorganized into more efficient, contiguous representations
- **Key properties:** Improves cache performance; simplifies allocation

---

## 13. Embedding and Similarity Search Algorithms (Conceptual Spaces)

### 13.1 Embedding Algorithms

#### Word2Vec (Skip-gram and CBOW)

- **What it does:** Learns word embeddings by predicting context words from a target word (Skip-gram) or a target word from context (CBOW)
- **Cognitive parallel:** Learning conceptual representations from patterns of co-occurrence — the contexts in which a concept appears define its meaning
- **Key result:** Vector arithmetic on meaning — vector("king") − vector("man") + vector("woman") ≈ vector("queen")
- **Training:** Stochastic gradient descent with negative sampling or hierarchical softmax

#### GloVe (Global Vectors for Word Representation)

- **What it does:** Learns embeddings from word co-occurrence statistics across a corpus, combining matrix factorization and local context windows
- **Key insight:** The ratio of co-occurrence probabilities encodes meaning — if P(ice|water)/P(ice|steam) is high, it captures the relationship between water, steam, and ice
- **Training:** Weighted least squares on the log of co-occurrence counts

#### FastText

- **What it does:** Extends Word2Vec by representing words as bags of character n-grams, allowing embeddings for unseen words
- **Cognitive parallel:** How you can infer the approximate meaning of an unfamiliar word from its parts — "unhappiness" from "un-," "happy," "-ness"
- **Key properties:** Handles morphology; produces embeddings for out-of-vocabulary words

### 13.2 Similarity Measures

#### Cosine Similarity

- **What it does:** Measures the cosine of the angle between two vectors: cos(θ) = (A · B) / (||A|| × ||B||)
- **Cognitive parallel:** Semantic similarity between distributed neural representations — concepts with similar patterns of activation are "close" in representational space
- **Range:** −1 (opposite) to +1 (identical direction)
- **Key properties:** Scale-invariant; widely used for text and embedding similarity

#### Euclidean Distance

- **What it does:** Straight-line distance between two points in n-dimensional space
- **Cognitive parallel:** "How different" two representations are in absolute terms
- **Key properties:** Sensitive to magnitude; most intuitive distance measure

### 13.3 Nearest Neighbor Search Algorithms

#### k-Nearest Neighbors (k-NN)

- **What it does:** Finds the k most similar items to a query by exhaustive comparison
- **Cognitive parallel:** Recalling a concept and activating its nearest semantic neighbors
- **Time complexity:** O(nd) for n items in d dimensions (brute force)
- **Classification variant:** Assigns the majority class among k nearest neighbors

#### Approximate Nearest Neighbor (ANN) Algorithms

These trade perfect accuracy for speed, enabling similarity search at scale:

**Locality-Sensitive Hashing (LSH):**
- Hashes similar items to the same bucket with high probability
- **Cognitive parallel:** Fast, approximate semantic retrieval — quickly narrowing to the right neighborhood of concepts
- **Time complexity:** Sub-linear query time
- **Key insight:** Use hash functions that preserve similarity — nearby points hash to the same bucket

**HNSW (Hierarchical Navigable Small World):**
- Builds a multi-layer graph; top layers have sparse, long-range connections for coarse search; bottom layers have dense, short-range connections for refinement
- **Cognitive parallel:** Hierarchical memory search — first narrowing to the right domain, then finding the specific concept
- **Time complexity:** O(log n) expected query time
- **Key properties:** State-of-the-art recall vs. speed trade-off

**FAISS (Facebook AI Similarity Search):**
- Library implementing multiple ANN strategies including IVF (inverted file index), PQ (product quantization), and HNSW
- **Key properties:** Billion-scale vector search; GPU-accelerated; production-grade

**Annoy (Approximate Nearest Neighbors Oh Yeah):**
- Uses random projection trees for fast ANN search
- **Key properties:** Memory-mapped; good for read-heavy workloads; used at Spotify

---

## 14. Reinforcement Learning Algorithms (Reward-Based Learning and Habit Formation)

Deeply relevant to procedural memory, the basal ganglia, and dopamine signaling.

### 14.1 Value-Based Methods

#### Q-Learning

- **What it does:** Learns the value of taking each action in each state through trial and error
- **Update rule:** Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') − Q(s,a)]
- **Cognitive parallel:** The temporal difference error [r + γ max Q(s',a') − Q(s,a)] has been directly linked to **dopamine signaling** in the basal ganglia — one of the most celebrated correspondences between an algorithm and a brain mechanism
- **Key properties:** Off-policy (can learn from hypothetical actions); converges to optimal policy with sufficient exploration

#### SARSA (State-Action-Reward-State-Action)

- **What it does:** Similar to Q-learning but updates based on the action actually taken rather than the best possible action
- **Update rule:** Q(s,a) ← Q(s,a) + α[r + γQ(s',a') − Q(s,a)] where a' is the action actually taken
- **Cognitive parallel:** Learning from what you actually do rather than what you could theoretically do — more conservative, risk-averse
- **Key properties:** On-policy; converges to the optimal policy for the current exploration strategy

#### Temporal Difference Learning (General Form)

- **What it does:** Learns predictions by comparing successive predictions rather than waiting for final outcomes
- **TD error:** δ = r + γV(s') − V(s)
- **Cognitive parallel:** Now widely accepted as a computational description of what dopamine neurons do — Wolfram Schultz's experiments showed dopamine neurons fire for unexpected rewards, are suppressed for unexpected reward omission, and show no response for fully predicted rewards — exactly matching TD error signals
- **Key properties:** Bootstrapping (learning predictions from predictions); faster than Monte Carlo methods; foundational concept in RL

### 14.2 Policy-Based Methods

#### REINFORCE (Williams, 1992)

- **What it does:** Directly optimizes the policy by computing the gradient of expected reward and adjusting action probabilities
- **Policy gradient:** ∇J(θ) = E[∇log π(a|s;θ) × R]
- **Cognitive parallel:** Directly adjusting behavioral probabilities based on outcomes — actions followed by reward become more likely
- **Key properties:** Can handle continuous action spaces; high variance (slow convergence)

#### Proximal Policy Optimization (PPO)

- **What it does:** Constrains policy updates to prevent too-large changes using a clipped objective function
- **Cognitive parallel:** Learning in measured steps — the brain doesn't completely overhaul its behavioral strategy after a single experience
- **Key properties:** Stable training; widely used (OpenAI's default algorithm); good balance of simplicity and performance

#### Actor-Critic Methods (A2C, A3C)

- **What it does:** Combines a policy network (actor — chooses actions) with a value network (critic — evaluates states)
- **Cognitive parallel:** The actor maps to the motor/executive system (choosing what to do); the critic maps to the evaluative system (assessing outcomes via dopamine signals)
- **A3C:** Asynchronous version with multiple parallel agents — like multiple brain systems simultaneously evaluating different scenarios

### 14.3 Model-Based Reinforcement Learning

- **What it does:** Builds an internal model of the environment (transition dynamics and rewards) and uses it to plan
- **Cognitive parallel:** The prefrontal cortex's role in mentally simulating outcomes before acting — imagining "what would happen if..." scenarios
- **Algorithms:** Dyna-Q (interleaves real experience with simulated experience), MCTS (Monte Carlo Tree Search), World Models
- **Key properties:** Sample-efficient (learns from fewer real experiences); can plan ahead; requires accurate model

---

## Summary: Complete Algorithm-to-Cognition Mapping

| Cognitive Process | Key Algorithms |
|---|---|
| Spreading activation | BFS, Dijkstra's |
| Focused reasoning | DFS, A*, backtracking |
| Conceptual importance | PageRank, betweenness centrality |
| Knowledge domains | Louvain, Girvan-Newman |
| Categorization | Binary search, tree traversals |
| Language processing | Trie search, prefix matching |
| Efficient coding | Huffman trees |
| Cue-based retrieval | Hashing, collision resolution |
| Recognition / familiarity | Bloom filters |
| Pattern completion | Hopfield networks, SDM |
| Cognitive interruption | Stack push/pop, call stack management |
| Attentional selection | Priority queues, heap operations |
| Working memory limits | Circular buffer overwrite |
| Episodic replay | Linked list traversal, skip lists |
| Neural computation | Matrix multiplication, SVD, PCA |
| Learning from error | Backpropagation, gradient descent |
| Robustness | Dropout, distributed representations |
| Visual processing | Convolution |
| Sequence learning | LSTM, GRU, BPTT |
| Flexible attention | Transformer self-attention |
| Skill automatization | NFA→DFA conversion, FSM minimization |
| Expert diagnosis | Decision trees (ID3, CART), random forests |
| Iterative refinement | Gradient boosted trees |
| Predictive models | VAEs, GANs, diffusion models |
| Efficient representation | LZ compression, arithmetic coding, delta encoding |
| Prediction errors | Diff algorithms, delta encoding |
| Memory consolidation | Cache algorithms (LRU, LFU, ARC) |
| Information retrieval | B-tree indexing, inverted indexing |
| Forgetting / pruning | Mark-and-sweep, reference counting, generational GC |
| Semantic similarity | Cosine similarity, Word2Vec, GloVe |
| Fast memory search | LSH, HNSW, FAISS |
| Reward-based learning | Q-learning, TD learning |
| Dopamine signaling | TD error |
| Behavioral adjustment | Policy gradient, PPO, actor-critic |
| Mental simulation | Model-based RL, planning algorithms |
