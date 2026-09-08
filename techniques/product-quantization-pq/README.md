# Product Quantization for ANN (Reference §47.84)

Jégou, Douze & Schmid (2011). Split each `d`-dimensional vector
into M sub-vectors of dim `d/M`; k-means each sub-space
independently to `K = 2^b` codewords. Each vector is stored as
M codebook indices — total `M · b` bits (typically 128 bits at
d=128).

Asymmetric distance approximation: precompute per-subspace
distance tables from the (uncompressed) query; scoring each stored
vector = M lookups + a sum. Under the FAISS umbrella of algorithms
that make billion-scale ANN feasible in RAM.

## Files

- `python/product_quantization_pq.py` — from-scratch PQ
  training + asymmetric distance computation. Demo (n=10 000, d=64,
  50 queries):
  - M= 4, K=256 →  32 bits/vec  Recall@10 = 0.09
  - M= 8, K=256 →  64 bits/vec  Recall@10 = 0.25
  - M=16, K=256 → 128 bits/vec  Recall@10 = 0.55.
  Full-precision baseline: 256 B/vec — PQ compresses ~16× at
  usable recall.
- `r/product_quantization_pq.R` — `RcppFaiss` (limited) (R);
  `faiss`, `scann`, from-scratch (Python).

## When to use

- **Billion-scale ANN** — RAM-bound corpus, recall > 0.5 acceptable.
- **Vector databases** — Milvus, Weaviate, Qdrant use PQ (and
  variants) under the hood.
- **Product recommendations** at retrieval time.
- **Image / document retrieval** with dense embeddings.

## When NOT to use

- **Small corpora** — brute-force or exact-tree indices are simpler.
- **When recall must exceed ~0.9** — combine with re-ranking on
  full-precision vectors (IVF-PQ + rerank).
- **Very low-dim vectors** (d ≤ 8) — kd-tree or brute-force beats PQ.

## Assumptions & caveats

- **Sub-space independence** assumed by PQ; add OPQ (Ge et al 2013)
  rotation to whiten dimensions first.
- **K, M trade-off** — larger K = more codewords per subspace,
  higher recall; larger M = more subspaces, finer quantisation.
- **Query is not compressed** in asymmetric variant — memory
  bound is on the corpus.
- **Coarse quantisation + IVF** (inverted-file list) usually
  paired with PQ in production (IVF-PQ).

## Related in this repo

- `hnsw-ann-search` — graph-based ANN alternative.
- `random-projections`, `matrix-completion-svt`, `sparse-pca`,
  `ica`, `nmf`, `pca`, `probabilistic-pca`, `kernel-pca` — dense
  embedding / dim-reduction toolkit.
- `min-hash-lsh`, `bloom-filter-membership`,
  `hyperloglog-cardinality` — probabilistic data structures.
- `feature-hashing`, `tfidf-bm25`, `sentence-similarity` — retrieval
  pipeline neighbours.

## Run

```
python techniques/product-quantization-pq/python/product_quantization_pq.py
Rscript techniques/product-quantization-pq/r/product_quantization_pq.R
```

**Refs:** Jégou, H., Douze, M. & Schmid, C. "Product quantization for nearest neighbor search." *IEEE TPAMI* 33(1): 117-128, 2011; Ge, T., He, K., Ke, Q. & Sun, J. "Optimized product quantization." *IEEE TPAMI* 36(4): 744-755, 2014.

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
