# Master Theorem Worked Examples — Social-Media-Radar

**Course:** CS / Software Engineering  
**Date:** 2026-03-18

---

## 1. The Master Theorem — Reference Statement

For a recurrence of the form

```
T(n) = a · T(n/b) + f(n),    a ≥ 1,  b > 1
```

define the **watershed exponent** `c* = log_b(a)`.  There are three cases:

| Case | Condition | Result |
|------|-----------|--------|
| **Case 1** | f(n) = O(n^(c*−ε)) for some ε > 0 | T(n) = Θ(n^c*) |
| **Case 2** | f(n) = Θ(n^c* · log^k n) for k ≥ 0 | T(n) = Θ(n^c* · log^(k+1) n) |
| **Case 3** | f(n) = Ω(n^(c*+ε)) and regularity holds | T(n) = Θ(f(n)) |

The Master Theorem applies **only** to divide-and-conquer recurrences where each subproblem has size exactly n/b.  It does **not** apply to recurrences of the form T(n) = T(n−1) + g(n); those require telescoping or the Akra-Bazzi method.

---

## 2. Example A — HNSW Index Construction

### 2.1 Background

The Hierarchical Navigable Small World (HNSW) index (`app/intelligence/hnsw_search.py`) is used by the candidate retriever to find approximate nearest neighbours among historical signal vectors.

During **construction**, each of n vectors is inserted into a layered graph.  Layer `ℓ` contains roughly `n · p^ℓ` nodes (p ≈ 1/e for the default `mL = 1/ln M`).  Inserting one vector at layer ℓ requires a greedy search of cost O(log n) (amortized over `ef_construction` candidates checked, but each search step is O(M·log M) comparisons).

The standard approximation for construction cost (Malkov & Yashunin 2020) gives:

```
T(n) = 2 · T(n/2) + O(n log n)
```

**Interpretation:** the construction of an index over n points is modelled as two independent sub-index constructions over n/2 points (one for the upper layers, one for the base layer), with an O(n log n) merging/linking pass.

### 2.2 Applying the Master Theorem

```
a = 2,  b = 2,  f(n) = c · n · log n
```

**Watershed exponent:**

```
c* = log_b(a) = log_2(2) = 1
```

**Classify f(n):**

```
f(n) = Θ(n^1 · log^1 n) = Θ(n^c* · log^1 n)
```

This matches **Case 2** with k = 1.

**Result:**

```
T(n) = Θ(n^c* · log^(k+1) n) = Θ(n · log² n)
```

**Verification:** The HNSW paper reports O(n log n) empirically for low-dimensional vectors; O(n log² n) is the worst-case upper bound used in the theoretical analysis.

### 2.3 Conclusion

> HNSW construction: **T(n) = Θ(n log² n)**  
> Master Theorem Case 2 applies with a=2, b=2, k=1.

---

## 3. Example B — Merge-Sort Pre-Sort for Signal Ranking

### 3.1 Background

Before presenting signals to analysts, the pipeline ranks by composite score (recency × topic-match × engagement).  A merge-sort pre-sort is performed on the raw signal list of n items.

Standard recursive merge sort:

```
T(n) = 2 · T(n/2) + c · n
```

**Interpretation:** sort two halves independently (each of size n/2), then merge in O(n) time.

### 3.2 Applying the Master Theorem

```
a = 2,  b = 2,  f(n) = c · n
```

**Watershed exponent:**

```
c* = log_2(2) = 1
```

**Classify f(n):**

```
f(n) = Θ(n) = Θ(n^1) = Θ(n^c*)
```

This matches **Case 2** with k = 0 (no log factor in f(n) itself).

**Result:**

```
T(n) = Θ(n^c* · log^(k+1) n) = Θ(n · log n)
```

**Verification:** This is the well-known result. For a list of 10⁶ signals, merge sort takes roughly 10⁶ · 20 = 2 × 10⁷ operations vs bubble sort's 10¹² — a 50,000× advantage.

### 3.3 Conclusion

> Merge-sort ranking pre-sort: **T(n) = Θ(n log n)**  
> Master Theorem Case 2 applies with a=2, b=2, k=0.

---

## 4. Example C — BFS: Why the Master Theorem Does NOT Apply

### 4.1 Background

BFS (benchmarked in `benchmark.py` and `report.md`) visits each of V vertices exactly once, processing d edges per vertex.  Its cost can be expressed as a recurrence:

```
T(V) = T(V−1) + O(d)
```

**This is NOT a divide-and-conquer recurrence.**

### 4.2 Why the Master Theorem Does Not Apply

The Master Theorem requires `T(n) = a · T(n/b) + f(n)` where b > 1, meaning each recursive call works on a *strictly smaller fraction* of the input.  BFS reduces the problem by one vertex per step (n → n−1), which corresponds to:

```
b = n/(n−1) → 1  as n → ∞
```

Since b → 1 (not a fixed constant > 1), the Master Theorem is inapplicable.  It is also inapplicable for a second reason: BFS is not divide-and-conquer at all — it processes vertices iteratively, not by splitting the graph into independent subgraphs.

### 4.3 Correct Solution: Telescoping

Unroll the recurrence T(V) = T(V−1) + c·d:

```
T(V) = T(V−1) + c·d
     = T(V−2) + c·d + c·d
     = T(V−3) + 3·c·d
     = ...
     = T(1)   + (V−1) · c·d
     = O(1)   + O(V·d)
     = O(V·d)
```

For a graph with E edges, the total degree sum is 2E, so O(V·d) = O(V + E).

**Result:** T(V) = **O(V + E)** — confirmed by the empirical linear fit in `report.md`.

### 4.4 Alternative: Akra-Bazzi

If BFS were applied recursively to sub-graphs of different sizes (e.g., via divide-and-conquer graph decomposition), the Akra-Bazzi method would apply.  Akra-Bazzi generalises the Master Theorem to recurrences of the form:

```
T(n) = Σᵢ aᵢ · T(bᵢ·n) + f(n)
```

and would yield Θ(n log n) for a balanced bisection with linear merge cost — but standard iterative BFS does not have this structure.

### 4.5 Conclusion

> BFS: **T(V) = O(V+E)** derived by telescoping (NOT Master Theorem).  
> The Master Theorem does not apply because BFS reduces by V−1, not V/b for fixed b.

---

## 5. Summary

| Algorithm | Recurrence | Method | Result |
|-----------|-----------|--------|--------|
| HNSW construction | T(n) = 2T(n/2) + O(n log n) | Master Theorem Case 2 | **Θ(n log² n)** |
| Merge-sort ranking | T(n) = 2T(n/2) + cn | Master Theorem Case 2 | **Θ(n log n)** |
| BFS traversal | T(V) = T(V−1) + O(d) | Telescoping (MT inapplicable) | **O(V+E)** |

---

## 6. Key Insight: When the Master Theorem Does NOT Apply

The Master Theorem fails in four common situations encountered in this codebase:

1. **Linear reduction (not geometric):** T(n) = T(n−1) + f(n) — use telescoping or substitution.
2. **Non-uniform subproblems:** T(n) = T(n/3) + T(2n/3) + f(n) — use Akra-Bazzi.
3. **Iterative algorithms:** BloomFilter, ReservoirSampler, ConfidenceCalibrator — no recursion; analyse the loop directly (counting operations per iteration).
4. **Non-recursive parallelism:** asyncio.gather dispatching n tasks — the cost is O(n·c) where c is per-task overhead; no recurrence applies.

Understanding these boundaries is essential for correctly classifying the complexity of algorithms in a real production codebase, where divide-and-conquer is the *exception*, not the rule.

