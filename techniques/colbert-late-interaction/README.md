# ColBERT — Late Interaction Retrieval (Reference §47.189)

Khattab & Zaharia (2020, SIGIR). Unlike DPR (one vector per doc),
ColBERT keeps a **vector per token** and scores at query time with
a **late-interaction MaxSim** operator:

    score(q, p) = Σ_{q_i ∈ q}  max_{p_j ∈ p} ⟨q_i, p_j⟩

Retrieval keeps token-level granularity (better recall on rare
terms) at the price of much larger index size. **ColBERTv2**
(Santhanam 2022) quantises tokens down to 32 bytes each.

## Files

- `python/colbert_late_interaction.py` — from-scratch ColBERT
  MaxSim vs DPR mean-pool on a 5-passage corpus with rare
  keyword queries ("quantum entanglement", "backpropagation",
  "recipe basil"):
  - Both methods get **3/3 Recall@1** on this small structured
    corpus; on larger, noisier corpora ColBERT's token-level
    matching typically wins by ~5-15 pt MRR.
- `r/colbert_late_interaction.R` — no R port; recommends
  `stanford-futuredata/ColBERT`, `bclavie/ragatouille`, `pyserini`.

## When to use

- **Open-domain QA** with technical / rare vocabulary — MaxSim
  finds the ONE token match even in noisy passages.
- **Zero-shot passage retrieval** at BEIR benchmark level —
  ColBERTv2 leads many datasets.
- **When index size is not the primary constraint**.

## When NOT to use

- **Extreme-scale corpora** (billions of passages) — index size
  is 10-100× larger than DPR.
- **Query-latency-critical** applications — MaxSim per-query is
  more expensive than DPR's single dot product.
- **When bi-encoder is enough** — for simple paraphrase retrieval.

## Assumptions & caveats

- **Index size scaling**: |vocab_per_doc| × dim × precision.
  ColBERTv2's 4-bit quantisation cuts index 8×.
- **PLAID** (Santhanam 2022) makes late-interaction retrieval
  scale to real-time serving via centroid-based pruning.
- **MaxSim** is asymmetric — training data must reflect
  query / passage token statistics.

## Related in this repo

- `dense-passage-retrieval-dpr` — single-vector bi-encoder
  cousin.
- `cross-encoder-reranker` — the deeper (but slower) alternative
  scoring model.
- `tfidf-bm25`, `reciprocal-rank-fusion` — sparse retrieval
  neighbours.
- `retrieval-augmented-generation`, `self-rag`,
  `context-compression-recomp` — downstream RAG pipeline.

## Run

```
python techniques/colbert-late-interaction/python/colbert_late_interaction.py
Rscript techniques/colbert-late-interaction/r/colbert_late_interaction.R
```

**Refs:** Khattab, O. & Zaharia, M. "ColBERT: Efficient and effective passage search via contextualized late interaction over BERT." *SIGIR*, 2020; Santhanam, K. et al. "ColBERTv2: Effective and efficient retrieval via lightweight late interaction." *NAACL*, 2022.

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
