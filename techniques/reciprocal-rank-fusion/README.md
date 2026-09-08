# Reciprocal Rank Fusion (Reference §47.131)

Cormack, Clarke & Buettcher (2009). Simple unsupervised rank
fusion:

    RRF(d) = Σ_{r ∈ R} 1 / (k + rank_r(d))

with k=60 by default (dampens tail contributions). Insensitive to
raw-score scales; treats each ranker as producing an ordinal
ranking only. State of the art for hybrid dense + sparse
retrieval (BM25 + vector) despite its simplicity.

## Files

- `python/reciprocal_rank_fusion.py` — from-scratch RRF + toy
  BM25-like vs dense-like rankers over 100 documents (10 truly
  relevant). Demo:
  - BM25-like NDCG@10 = 0.908
  - Dense-like NDCG@10 = 0.897
  - RRF fused (k=60)   = 0.870
  - RRF fused (k=1)    = **0.936** (small k more aggressive).
- `r/reciprocal_rank_fusion.R` — no first-class R port;
  `ranx`, `pytrec_eval`, `pyserini` (Python).

## When to use

- **Hybrid retrieval** — combine BM25 + dense vector search.
- **Ensembles of rankers** without training a supervised model.
- **When ranker scores are on incomparable scales**.

## When NOT to use

- **When you have labelled relevance data** — LambdaMART / neural
  LTR trained on it will win.
- **When one ranker is much stronger** — fusion may hurt; consider
  reranking instead.
- **Very short rankings** where individual position matters
  precisely — the k+rank denominator dampens rank-1 vs rank-2.

## Assumptions & caveats

- **k choice** — 60 canonical; sensitivity moderate.
- **Deduplication** across rankers essential.
- **Truncation** — fuse only the top-N of each ranker at retrieval
  time.
- **Bias**: rankers correlated in errors don't gain from fusion.

## Related in this repo

- `learning-to-rank-lambdamart` — supervised ranker often paired
  with RRF for hybrid stacks.
- `matrix-factorization-als-recsys`,
  `neural-collaborative-filtering`, `tfidf-bm25`,
  `sentence-similarity` — retrieval components.
- `hnsw-ann-search`, `min-hash-lsh`,
  `product-quantization-pq` — ANN backends.
- `discrimination-calibration`, `youden-optimal-cutpoint`,
  `f1-optimal-threshold` — evaluation cousins.

## Run

```
python techniques/reciprocal-rank-fusion/python/reciprocal_rank_fusion.py
Rscript techniques/reciprocal-rank-fusion/r/reciprocal_rank_fusion.R
```

**Refs:** Cormack, G.V., Clarke, C.L.A. & Büttcher, S. "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods." *SIGIR*, 2009.

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
