# Self-RAG (Reference §47.192)

Asai, Wu, Wang, Sil & Hajishirzi (2024, ICLR). The LM learns
**reflection tokens** that let it:

- **Retrieve?**  yes / no  (decide when to fetch context)
- **IsRel(d, q)?**  yes / partial / irrelevant per passage
- **IsSup(gen, d)?**  supported / partially / not-supported
- **IsUse(gen)**  overall utility 1-5

At inference the model generates candidate answers with different
retrieved subsets, scored by the reflection tokens, and picks the
best. Beats standard RAG on ambiguous or misleading contexts.

## Files

- `python/self_rag.py` — toy Self-RAG pipeline with rule-based
  IsRel / IsSup / IsUse tokens on a 5-passage corpus with a
  distractor passage. Both plain-RAG and Self-RAG achieve 3/3 on
  the demo queries; real Self-RAG's benefit shows on ambiguous
  or misleading contexts and long-tail queries.
- `r/self_rag.R` — no R port; recommends `AkariAsai/self-rag`,
  `langchain` Self-RAG chain, `llama-index` self-critique.

## When to use

- **RAG over noisy / ambiguous corpora** — critic tokens filter
  out irrelevant passages.
- **When retrieval quality varies per query** — the "should I
  retrieve at all?" decision saves latency on trivial questions.
- **Domain adaptation** — fine-tune reflection tokens on your
  domain.

## When NOT to use

- **Small, high-precision corpora** — plain RAG is simpler and
  as good.
- **Latency-sensitive** applications — multiple generations per
  query.
- **When base LM is very small** — reflection-token training
  requires a competent base.

## Assumptions & caveats

- **Requires fine-tuning** — reflection tokens are new vocabulary
  the base LM must learn.
- **Score aggregation** — weighted combination of IsRel + IsSup +
  IsUse; tune per domain.
- **Not a hallucination guarantee** — the critic can itself be
  wrong.
- **Compute cost** — 3-5× plain RAG per query (multiple candidate
  generations).

## Related in this repo

- `retrieval-augmented-generation` — the standard RAG baseline.
- `hyde-hypothetical-doc`, `context-compression-recomp`,
  `cross-encoder-reranker` — orthogonal RAG improvements.
- `constitutional-ai`, `rlhf-preferences`,
  `dpo-direct-preference-optimization` — related self-critique
  training methods.
- `tree-of-thoughts-reasoning` — search-based reasoning cousin.

## Run

```
python techniques/self-rag/python/self_rag.py
Rscript techniques/self-rag/r/self_rag.R
```

**Refs:** Asai, A., Wu, Z., Wang, Y., Sil, A. & Hajishirzi, H. "Self-RAG: Learning to retrieve, generate, and critique through self-reflection." *ICLR*, 2024.

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
