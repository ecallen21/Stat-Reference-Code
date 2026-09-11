# Cross-Encoder Reranker (Reference §47.190)

Nogueira & Cho (2019); Wang et al. (2020, MiniLM). Bi-encoder
retrieval (DPR, ColBERT) is fast but shallow: q and p are encoded
INDEPENDENTLY. Cross-encoders re-rank the top-K by feeding
[q; SEP; p] through a single Transformer:

    score(q, p) = MLP(BERT([q; SEP; p]))

Much more accurate (full attention between q and p) but O(K)
forward passes per query, so used only for reranking the top ~100
from a cheap first-stage retriever.

## Files

- `python/cross_encoder_reranker.py` — bi-encoder first stage,
  cross-encoder rerank. Toy 5-query / 20-passage corpus with kw-
  based golden pairs:
  - Bi-encoder Recall@1 = **5/5**, cross-encoder Recall@1 = **5/5**
    on this simplified task; on realistic noisy corpora the
    cross-encoder typically adds 5-15 pt MRR@10.
- `r/cross_encoder_reranker.R` — no R port; recommends
  `sentence-transformers.CrossEncoder`, MonoT5 / MiniLM checkpoints
  on HF Hub.

## When to use

- **Second-stage reranking** of top-100 from any first-stage
  retriever.
- **When quality > latency** — batch scoring, offline eval.
- **Zero-shot** with strong pretrained rerankers (bge-reranker,
  cross-encoder/ms-marco-*).

## When NOT to use

- **First-stage retrieval** over large corpora — O(N) per query
  is too slow.
- **Latency-critical serving** — cross-encoder adds ~5-20 ms per
  candidate.
- **When bi-encoder is already saturated on the task**.

## Assumptions & caveats

- **Sequence-length limits** — [q; SEP; p] must fit; long passages
  are truncated / chunked.
- **Point-wise vs pair-wise loss** — MonoT5 (pointwise), DuoT5
  (pairwise) trade quality vs speed.
- **Training data** — MS MARCO / TREC DL for open-domain; domain
  fine-tuning is usually a large win.
- **Score calibration** — cross-encoder logits are not
  probabilities without additional calibration.

## Related in this repo

- `dense-passage-retrieval-dpr`, `colbert-late-interaction` —
  first-stage retrievers reranker sits on top of.
- `tfidf-bm25`, `reciprocal-rank-fusion` — sparse first-stage
  cousins.
- `retrieval-augmented-generation`, `self-rag`,
  `hyde-hypothetical-doc` — RAG pipeline neighbours.
- `learning-to-rank-lambdamart` — related ranking framework.

## Run

```
python techniques/cross-encoder-reranker/python/cross_encoder_reranker.py
Rscript techniques/cross-encoder-reranker/r/cross_encoder_reranker.R
```

**Refs:** Nogueira, R. & Cho, K. "Passage re-ranking with BERT." *arXiv:1901.04085*, 2019; Wang, W. et al. "MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers." *NeurIPS*, 2020; Nogueira, R. et al. "Document ranking with a pretrained sequence-to-sequence model (MonoT5)." *EMNLP*, 2020.

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
