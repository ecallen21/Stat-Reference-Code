# Dense Passage Retrieval — DPR (Reference §47.188)

Karpukhin et al. (2020, EMNLP). Two BERT encoders (question
encoder E_q, passage encoder E_p) trained with **in-batch
contrastive loss**:

    sim(q, p) = E_q(q) · E_p(p)

Positive: (question, gold passage); negatives: other passages in
the batch + optional hard negatives from BM25. Retrieval = FAISS
inner-product search over dense passage embeddings.

Beats sparse BM25 by ~10 pt Recall@20 on Natural Questions.

## Files

- `python/dense_passage_retrieval_dpr.py` — toy DPR with mean-of-
  embeddings encoders on 8 (question, passage) pairs:
  - **DPR Recall@1 = 1.00** (perfect on training pairs).
  - BM25 baseline Recall@1 = 0.75.
  - DPR Recall@3 = BM25 Recall@3 = 1.00.
- `r/dense_passage_retrieval_dpr.R` — no R port; recommends
  `facebookresearch/DPR`, `sentence-transformers`, `haystack`.

## When to use

- **Open-domain QA / semantic search** — sparse-only (BM25) is
  brittle on paraphrase.
- **Zero-shot** with pretrained encoders (BGE, GTE, MPNet).
- **Fine-tuning** on your domain's (query, doc) pairs is a large
  quality win.

## When NOT to use

- **Exact-match keyword queries** (product codes, SKUs) — BM25 is
  simpler and better.
- **When training data is scarce and off-the-shelf embeddings
  underperform**.
- **Very large corpora without ANN indexing** — brute-force cosine
  is O(N).

## Assumptions & caveats

- **In-batch negatives** need large batch (paper uses 128) to work
  well.
- **Hard negatives** from BM25 mining substantially improve
  Recall@1.
- **Query/doc encoder asymmetry** matters — tie them for symmetric
  tasks, keep separate for QA.
- **FAISS index type** — flat for accuracy, IVF/HNSW for speed.

## Related in this repo

- `colbert-late-interaction` — token-level late-interaction
  cousin.
- `cross-encoder-reranker` — 2-stage rerank companion.
- `hyde-hypothetical-doc` — query-side augmentation.
- `retrieval-augmented-generation` — the downstream LLM pipeline.
- `reciprocal-rank-fusion`, `tfidf-bm25` — sparse / hybrid
  retrieval siblings.

## Run

```
python techniques/dense-passage-retrieval-dpr/python/dense_passage_retrieval_dpr.py
Rscript techniques/dense-passage-retrieval-dpr/r/dense_passage_retrieval_dpr.R
```

**Refs:** Karpukhin, V. et al. "Dense passage retrieval for open-domain question answering." *EMNLP*, 2020; Xiong, L. et al. "Approximate nearest neighbor negative contrastive learning for dense text retrieval (ANCE)." *ICLR*, 2021.

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
