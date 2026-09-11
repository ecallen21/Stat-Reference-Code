# HyDE — Hypothetical Document Embeddings (Reference §47.191)

Gao, Ma, Lin & Callan (2022, ACL 2023). To reduce the
QUERY-DOCUMENT distribution mismatch in dense retrieval, HyDE:

    1. Use an LM to generate a hypothetical passage h from the
       query q  (h ~ LM("write a passage that answers: {q}")).
    2. Embed h (not q) with the passage encoder.
    3. Retrieve nearest passages to embed(h).

The generated h shares the document distribution so retrieval
scores align better. Fully **zero-shot**: no supervised
fine-tuning of the encoder needed.

## Files

- `python/hyde_hypothetical_doc.py` — toy HyDE with a 5-passage
  corpus and 5 paraphrased queries (e.g. "how many people live in
  tokyo"):
  - **Plain-DPR Recall@1 = 0/5** (query-doc lexical mismatch)
  - **HyDE Recall@1 = 3/5** (generated passage aligns with corpus
    distribution).
- `r/hyde_hypothetical_doc.R` — no R port; recommends
  `llama-index.HyDEQueryTransform`, `langchain` HyDE integration.

## When to use

- **Zero-shot dense retrieval** where the encoder is not
  fine-tuned on the domain.
- **Short / colloquial queries** that don't match the
  document-corpus style.
- **When an LM is cheap to call** — HyDE adds one generation per
  query.

## When NOT to use

- **When encoder is fine-tuned on (query, doc) pairs** — HyDE's
  benefit shrinks.
- **Ultra-latency-critical** — the extra LM call adds ~100 ms.
- **Adversarial / underspecified queries** — hallucinated
  hypothetical can mislead.

## Assumptions & caveats

- **Hallucination is OK for retrieval** — the generated passage
  just needs to be topically near real passages.
- **Multiple hypotheticals** (average their embeddings) is
  usually better than one.
- **Query preservation** — some impls also blend embed(q) with
  embed(h).
- **LM cost** — cheaper with a small LM (~1 B params) than the
  generation LLM.

## Related in this repo

- `dense-passage-retrieval-dpr`, `colbert-late-interaction` —
  the retrievers HyDE augments.
- `cross-encoder-reranker` — orthogonal 2nd-stage reranker.
- `self-rag`, `context-compression-recomp`,
  `retrieval-augmented-generation` — RAG pipeline neighbours.

## Run

```
python techniques/hyde-hypothetical-doc/python/hyde_hypothetical_doc.py
Rscript techniques/hyde-hypothetical-doc/r/hyde_hypothetical_doc.R
```

**Refs:** Gao, L., Ma, X., Lin, J. & Callan, J. "Precise zero-shot dense retrieval without relevance labels (HyDE)." *ACL*, 2023.

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
