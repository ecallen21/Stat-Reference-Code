# HNSW Approximate NN Search (Reference §47.85)

Malkov & Yashunin (2018). Multi-layer Navigable Small World
proximity graph:

    * Layer 0: all data, dense local neighbours.
    * Higher layers: exponentially-sparser random subsets with
      long-range edges.

Search enters at the top layer with a greedy descent, keeping
`ef` nearest candidates at each hop. Average O(log n) query
complexity. HNSW leads ANN-Benchmarks for recall > 0.9 across
most datasets.

## Files

- `python/hnsw_ann_search.py` — uses `hnswlib` when installed;
  otherwise falls back to a from-scratch single-layer k-NN
  proximity graph with heap-based greedy search. Demo (n=50 000,
  d=32, 100 queries):
  - brute-force baseline 0.28 s for k=10
  - fallback graph ef=20  → 0.11 s   Recall@10 = 0.50
  - fallback graph ef=100 → 0.33 s   Recall@10 = 0.81
  - fallback graph ef=400 → 1.00 s   Recall@10 = 0.93.
- `r/hnsw_ann_search.R` — `RcppHNSW::hnsw_build/search` (R);
  `hnswlib`, `faiss`, `scann`, from-scratch (Python).

## When to use

- **Billion-scale ANN with recall > 0.9** — HNSW's sweet spot.
- **Latency-sensitive retrieval** — sub-millisecond queries at
  d ~ 100 typical.
- **Vector databases** — Milvus, Weaviate, Qdrant, pgvector all
  ship HNSW.
- **Semantic search over embeddings** (LLM RAG, image retrieval).

## When NOT to use

- **Very high recall requirement (~1.0)** — exact search or IVF-PQ
  + rerank.
- **Highly memory-constrained** — HNSW graph adds ~O(M · n)
  overhead; PQ is more compact.
- **Frequent updates / deletions** — HNSW supports both but is
  optimised for read-heavy workloads.
- **Very low-dim data** — kd-tree beats HNSW below d ~ 10.

## Assumptions & caveats

- **M** (max neighbours per node) trades index size for recall;
  M=16–32 typical.
- **efConstruction** vs **ef** at query — larger = higher recall,
  slower build / query.
- **Deterministic on fixed seed** for construction; queries stochastic
  in tie-breaking.
- **Deletions** mark as tombstones — periodic rebuild recommended.

## Related in this repo

- `product-quantization-pq` — memory-compressed alternative
  (often combined as IVF-PQ + HNSW re-rank).
- `min-hash-lsh`, `random-projections` — probabilistic ANN cousins.
- `random-fourier-features`, `nadaraya-watson-kernel-regression`,
  `kernel-density-estimation` — kernel-search-adjacent.
- `retrieval-augmented-generation`, `sentence-similarity`,
  `word-embeddings`, `tfidf-bm25` — retrieval pipelines using
  HNSW under the hood.

## Run

```
python techniques/hnsw-ann-search/python/hnsw_ann_search.py
Rscript techniques/hnsw-ann-search/r/hnsw_ann_search.R
```

**Refs:** Malkov, Y.A. & Yashunin, D.A. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." *IEEE TPAMI* 42(4): 824-836, 2020 (arXiv 2016).

---

## Author

Elisabeth F. Callen, Ph.D., PStat®
Biostatistician and applied health data researcher

[LinkedIn](https://www.linkedin.com/in/your-profile) · [ORCID](https://orcid.org/your-id) · elisabeth.f.callen@gmail.com

## Acknowledgments

**AI tooling.** This codebase was developed with the support of AI coding assistants (Claude Code). Methodology, statistical approach, validation logic, and interpretation of results are my own. AI tooling was used to accelerate code drafting, refactor for readability, and assist with documentation. All code was reviewed, tested, and validated against expected outputs before committing.

No protected health information was ever provided to AI coding assistants. All development and testing was conducted against synthetic data.

## License

[MIT](../../LICENSE)
