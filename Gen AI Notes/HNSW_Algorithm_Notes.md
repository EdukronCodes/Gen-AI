# HNSW Algorithm — Complete Notes

**Hierarchical Navigable Small World** — Approximate Nearest Neighbor (ANN) search for vector similarity.

---

## Table of Contents

1. [What is HNSW?](#1-what-is-hnsw)
2. [Why HNSW Exists](#2-why-hnsw-exists)
3. [Core Idea & Structure](#3-core-idea--structure)
4. [Key Parameters](#4-key-parameters)
5. [First Step: Random Level Assignment](#5-first-step-random-level-assignment)
6. [Understanding the Level Output](#6-understanding-the-level-output)
7. [Number of Levels & O(log N)](#7-number-of-levels--olog-n)
8. [Search Walkthrough (10 Sentences)](#8-search-walkthrough-10-sentences)
9. [Local Search (efSearch)](#9-local-search-efsearch)
10. [Summary & Interview Notes](#10-summary--interview-notes)

---

## 1. What is HNSW?

HNSW is an **approximate nearest neighbor (ANN)** search algorithm used in **vector databases** to efficiently find similar vectors in **high-dimensional spaces**.

**Common use cases:**

- Vector databases (Pinecone, Milvus, Weaviate, FAISS)
- Semantic search
- Recommendation systems
- RAG (Retrieval-Augmented Generation)
- Image, text, and embedding similarity search

---

## 2. Why HNSW Exists

### Problem with Exact Search

- Exact KNN requires comparing the query with **all vectors**.
- Time complexity: **O(N × D)**.
- Not scalable for millions/billions of vectors.

### Traditional Indexing Limitations

| Method   | Limitation              |
|----------|-------------------------|
| KD-Tree  | Fails in high dimensions |
| Ball Tree| Poor scalability        |
| LSH      | Lower accuracy          |
| Flat Index | Too slow              |

**HNSW** solves this using **graph-based navigation + hierarchy**.

---

## 3. Core Idea & Structure

### Intuition

- Based on **Small World Graphs**: each node has few neighbors; any node is reachable in **logarithmic steps**.
- HNSW adds a **hierarchy of graphs**:
  - **Top layers** → sparse, long-range connections (fast jumps).
  - **Bottom layer** → dense, local connections (precision).

### Search Flow

1. Start from the **top layer**.
2. Greedily move toward the query.
3. Descend layer by layer.
4. At the bottom layer, perform a local search and return top-k.

### Structure (Layers)

```
Level 3 (Very Sparse)     ●────────●
                                  │
Level 2                    ●───●───●
                                  │
Level 1                    ●──●──●──●──●
                                  │
Level 0 (Dense – All)      ●─●─●─●─●─●─●─●
```

- **Level 0**: Contains all vectors.
- **Higher levels**: Fewer vectors, used for fast navigation.

---

## 4. Key Parameters

| Parameter       | Meaning                          |
|----------------|-----------------------------------|
| `M`            | Max neighbors per node            |
| `efConstruction` | Accuracy vs build speed         |
| `efSearch`     | Accuracy vs query latency         |
| `L` / levels   | Emerges from random assignment    |

---

## 5. First Step: Random Level Assignment

The **first step** in HNSW when inserting a vector is:

> **Assign a random maximum level to the node.**

- Every node is added to **Level 0**.
- With probability `p`, the node is also promoted to the next level, and so on.
- This gives **few nodes at top layers**, **many at bottom** → logarithmic search.

### Mathematical Intuition

\[
P(\text{level} \geq \ell) = e^{-\ell/m}
\]

So higher levels have exponentially fewer nodes.

### Python + NumPy Code (First Step Only)

```python
import numpy as np

def assign_hnsw_level(p=0.5, max_level=10):
    """
    Assign a random maximum level to a node in HNSW.

    Parameters:
        p         : probability of going to the next level
        max_level : safety cap to avoid infinite loops

    Returns:
        level (int): maximum level assigned to the node
    """
    level = 0
    while np.random.rand() < p and level < max_level:
        level += 1
    return level


# Example: Assign levels to 10 nodes
np.random.seed(42)
num_nodes = 10
levels = np.array([assign_hnsw_level(p=0.5) for _ in range(num_nodes)])

print("Assigned HNSW Levels:")
print(levels)
```

**Typical output:**

```
Assigned HNSW Levels:
[0 1 0 2 0 0 1 0 3 0]
```

---

## 6. Understanding the Level Output

### What the Array Means

```text
Assigned HNSW Levels:
[0 1 0 2 0 0 1 0 3 0]
```

- **Index** = node (vector) ID.
- **Value** = highest layer where that node exists.
- Every node exists in **Level 0**; higher levels are optional.

### Node-by-Node Interpretation

| Node ID | Assigned Level | Meaning                          |
|---------|----------------|-----------------------------------|
| 0       | 0              | Appears only in Level 0           |
| 1       | 1              | Appears in Levels 0 and 1         |
| 2       | 0              | Appears only in Level 0           |
| 3       | 2              | Appears in Levels 0, 1, and 2     |
| 4       | 0              | Appears only in Level 0           |
| 5       | 0              | Appears only in Level 0           |
| 6       | 1              | Appears in Levels 0 and 1         |
| 7       | 0              | Appears only in Level 0           |
| 8       | 3              | Appears in Levels 0, 1, 2, and 3   |
| 9       | 0              | Appears only in Level 0           |

### Layer-Wise Distribution

| Level | Nodes in This Level |
|-------|----------------------|
| Level 3 | [8]                |
| Level 2 | [3, 8]             |
| Level 1 | [1, 3, 6, 8]       |
| Level 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |

- **Level 0** → dense (all nodes).
- **Higher levels** → sparse; **Node 8** is the natural **entry point** for search.

### Intuition

- **Top layers** = highways (long jumps).
- **Middle layers** = main roads.
- **Level 0** = local streets (precise search).

---

## 7. Number of Levels & O(log N)

### How Many Levels?

HNSW **does not fix** the number of levels. It **emerges** from random level assignment:

```text
level = 0
while random() < p:
    level += 1
```

- Most nodes stop at level 0.
- Very few reach high levels.

### Expected Nodes per Level

With \(N\) nodes and promotion probability \(p \approx 0.5\):

| Level | Expected number of nodes |
|-------|---------------------------|
| 0     | \(N\)                     |
| 1     | \(N \cdot p\)             |
| 2     | \(N \cdot p^2\)           |
| \(k\) | \(N \cdot p^k\)           |

So each level has **exponentially fewer** nodes.

### Total Number of Levels

The highest level \(L\) roughly satisfies \(N \cdot p^L \approx 1\), so:

\[
L \approx \log_{1/p}(N)
\]

For \(p = 0.5\): \(L \approx \log_2(N)\). So **number of levels is O(log N)**.

### Why Search is O(log N)

1. **Top layer** has very few nodes → fast greedy step.
2. **Each lower layer** refines the position; work per layer is bounded.
3. **Total layers** ≈ \(\log N\).
4. **Total search cost** ≈ (work per layer) × (number of layers) ≈ **O(log N)**.

### Interview-Ready Explanation

> "HNSW does not fix the number of levels. Each node is randomly promoted using a geometric distribution, so higher layers have exponentially fewer nodes. That gives about \(\log_2(N)\) layers, and with constant work per layer, search complexity is O(log N)."

---

## 8. Search Walkthrough (10 Sentences)

### Example Data: 10 Sentences

| ID | Sentence |
|----|----------|
| 0 | Machine learning models learn patterns from data. |
| 1 | Deep learning uses neural networks with many layers. |
| 2 | Azure Databricks is used for large-scale data processing. |
| 3 | Azure Data Factory orchestrates data pipelines. |
| 4 | PySpark helps process big data efficiently. |
| 5 | The football world cup is watched globally. |
| 6 | Cricket is one of the most popular sports in India. |
| 7 | The final match was played in a packed stadium. |
| 8 | The team celebrated after winning the championship. |
| 9 | Fans cheered loudly during the match. |

### Assigned HNSW Levels

```text
[0, 1, 0, 2, 0, 0, 1, 0, 3, 0]
```

**Layer-wise:**

- Level 3: [8]
- Level 2: [3, 8]
- Level 1: [1, 3, 6, 8]
- Level 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  
**Entry point = Node 8** (highest level).

### Search Query

**Query:** `"azure data pipelines"`  
**Goal:** Find the most semantically similar sentences.

### HNSW Search Flow (Layer by Layer)

#### Step 1: Top Layer (Level 3)

- **Nodes:** [8]
- **Node 8:** "The team celebrated after winning the championship."
- **Similarity with query** → low (sports vs data).
- **Action:** No better node; stay at 8 and move down.

#### Step 2: Level 2

- **Nodes:** [3, 8]

| Node | Sentence                              | Similarity |
|------|----------------------------------------|------------|
| 8    | Team celebrated after winning championship | Low    |
| 3    | Azure Data Factory orchestrates data pipelines | High |

- **Action:** Greedy move **8 → 3**. Best at Level 2 = Node 3.

#### Step 3: Level 1

- **Neighbors:** [1, 3, 6, 8]

| Node | Sentence                              | Similarity |
|------|----------------------------------------|------------|
| 1    | Deep learning uses neural networks     | Medium     |
| 3    | Azure Data Factory orchestrates data pipelines | Highest |
| 6    | Cricket is popular in India            | Low        |
| 8    | Team celebrated championship           | Low        |

- **Action:** Stay at **Node 3**. Move down to Level 0.

#### Step 4: Level 0 (Final — Dense Search)

- **All nodes:** [0–9].
- HNSW runs a **local search (efSearch)** around Node 3 (see next section).

**Ranked candidates (conceptually):**

| Rank | Node | Sentence |
|------|------|----------|
| 1    | 3    | Azure Data Factory orchestrates data pipelines |
| 2    | 2    | Azure Databricks is used for large-scale data processing |
| 3    | 4    | PySpark helps process big data efficiently |
| 4    | 0    | Machine learning models learn patterns from data |
| —    | 5–9  | Sports-related (lower similarity) |

### Final Top-K (e.g. k = 3)

1. Azure Data Factory orchestrates data pipelines  
2. Azure Databricks is used for large-scale data processing  
3. PySpark helps process big data efficiently  

### Visual Summary

```text
Level 3:      [8]        →  jump (sports node)
                  │
Level 2:   [3, 8]         →  correct cluster (Node 3)
                  │
Level 1: [1, 3, 6, 8]     →  refine (stay at 3)
                  │
Level 0: [all nodes]      →  local search → top-k
```

### What This Demonstrates

- High layers quickly skip the irrelevant (sports) cluster.
- Lower layers refine the search.
- No full scan over all 10 sentences.
- Search cost is logarithmic in the size of the index.

---

## 9. Local Search (efSearch)

After **greedy descent**, HNSW has an entry node at **Level 0** (in our example, **Node 3**). The next step is:

> **Perform a local search (efSearch) around that node.**

### What is efSearch?

- **efSearch** = how many **candidate nodes** HNSW is allowed to explore at the bottom layer (Level 0).
- **Larger efSearch** → higher accuracy/recall, slower.
- **Smaller efSearch** → faster, lower recall.
- Typical range in production: **50–200**.

### Why Local Search?

- Greedy descent gets **close** to the true neighbors.
- Local search **refines** by exploring the neighborhood.
- Together: **fast + accurate** ANN.

### Starting Point (Our Example)

- **Entry node at Level 0:** Node 3  
  ("Azure Data Factory orchestrates data pipelines")
- **efSearch = 4** (for illustration).

### Neighbors of Node 3 at Level 0 (Example)

Assume Node 3 is connected to:

```text
Node 3 neighbors (Level 0): [0, 2, 4, 1]
```

| Node | Sentence |
|------|----------|
| 0 | Machine learning models learn patterns from data |
| 2 | Azure Databricks is used for large-scale data processing |
| 4 | PySpark helps process big data efficiently |
| 1 | Deep learning uses neural networks |

### efSearch Exploration Process

**Step 1: Initialize**

```text
Candidates = [3]
Visited = {3}
```

**Step 2: Expand (until efSearch limit)**

- Evaluate query `"azure data pipelines"` vs neighbors.
- Add best unexplored neighbors to candidates.

**Step 3: Similarity (conceptual)**

| Node | Similarity (relative) |
|------|------------------------|
| 3    | Highest                |
| 2    | High                   |
| 4    | High                   |
| 0    | Medium                 |
| 1    | Low                    |

**Candidates after expansion (efSearch = 4):**

```text
Candidates = [3, 2, 4, 0]
```

**Step 4: Rank by similarity**

1. Node 3 — Azure Data Factory orchestrates data pipelines  
2. Node 2 — Azure Databricks is used for large-scale data processing  
3. Node 4 — PySpark helps process big data efficiently  
4. Node 0 — Machine learning models learn patterns from data  

**Step 5: Top-k**

If **k = 3**:  
Top-3 = **[3, 2, 4]**.

### Why This Is Efficient

- Search is limited to a **small neighborhood**.
- No full scan of all 10 nodes.
- **efSearch** caps the work at the bottom layer.
- Accuracy comes from exploring **enough** candidates, not all.

### Intuition

- **efSearch** ≈ “How many nearby houses I check after reaching the right street.”
- You don’t search the whole city — only the relevant area.

### Interview-Ready Explanation

> "After reaching the best entry node at the lowest layer, HNSW does a local search controlled by efSearch. It explores a limited number of nearby nodes to refine results, balancing accuracy and latency. Greedy descent gets you close; efSearch makes the answer correct."

**Takeaway:**  
**Greedy descent → gets close.** **efSearch → makes it correct.** Together they give **fast + accurate** ANN search.

---

## 10. Summary & Interview Notes

### One-Line Summary

- **HNSW** = graph + hierarchy for approximate nearest neighbor search in **O(log N)** time.

### Key Facts

| Topic | Point |
|-------|--------|
| First step | Random level assignment (geometric distribution). |
| Level output | Index = node ID, value = max level; Level 0 = all nodes. |
| Number of levels | Not fixed; emerges as ≈ \(\log_2(N)\) for \(p=0.5\). |
| Search complexity | O(log N) due to logarithmic number of layers and bounded work per layer. |
| At Level 0 | Local search with **efSearch** candidates for refinement. |

### Real-World Analogy

- **Top layers** = highways (fast, long jumps).  
- **Middle layers** = main roads.  
- **Level 0** = local streets (precise search).  
- **efSearch** = how many nearby houses you check on that street.

### When to Use HNSW

- **Good for:** High-dimensional embeddings, semantic search, RAG, recommendations, real-time ANN.
- **Avoid when:** Dataset is very small or you need **exact** nearest neighbors.

---

*End of HNSW Algorithm Notes.*
