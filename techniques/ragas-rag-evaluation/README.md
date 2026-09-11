# RAGAS — Retrieval-Augmented Generation Assessment (Reference §47.218)

Es, James, Espinosa-Anke & Schockaert (2024). Reference-free
metrics for RAG pipelines:

- **Faithfulness** = fraction of answer claims supported by
  retrieved context.
- **Answer relevance** = cosine(embed(gen answer), embed(query)).
- **Context precision** = signal-to-noise of retrieved chunks.
- **Context recall** = fraction of gold-answer claims present in
  retrieved context.

Uses an LM to extract claims + judge support; no gold answers
required for the first two.

## Files

- `python/ragas_rag_evaluation.py` — toy RAGAS on a 3-response
  Q&A ("When was the Eiffel Tower built and how tall?"):
  - Good answer: **faithfulness 1.00**, relevance 0.59.
  - Off-topic answer: faithfulness 0.00, relevance 0.26.
  - Toy keyword-overlap claim-support is a limitation — real
    RAGAS uses an LM for claim extraction and support judging.
- `r/ragas_rag_evaluation.R` — no R port; recommends `ragas`,
  `deepeval`, `trulens-eval`.

## When to use

- **RAG-pipeline evaluation** without gold-labelled QA pairs.
- **A / B testing** retrievers, rerankers, prompt templates.
- **Regression testing** — catch RAG drift after model updates.

## When NOT to use

- **When gold answers exist** — exact-match / F1 is cheaper and
  less biased.
- **Very short answers** — claim extraction is unstable on
  1-sentence responses.
- **Domain-specialised** where the judge LM lacks vocabulary
  (medical, legal without domain adapt).

## Assumptions & caveats

- **Judge LM quality** dominates — GPT-4 / Claude 3.5 typically;
  small LMs give noisy scores.
- **Claim decomposition** — the atomic-claim granularity shapes
  the metric.
- **Human correlation** ~ 0.65-0.80 depending on metric and
  domain.
- **Retrieval / generation coupling** — RAGAS can mis-attribute
  bad answers to bad retrieval (or vice versa).

## Related in this repo

- `retrieval-augmented-generation`, `self-rag`,
  `hyde-hypothetical-doc`, `context-compression-recomp` — RAG
  pipeline neighbours.
- `dense-passage-retrieval-dpr`, `colbert-late-interaction`,
  `cross-encoder-reranker` — retriever components RAGAS
  benchmarks.
- `llm-as-a-judge`, `mmlu-benchmark-eval` — sibling LM
  evaluation methods.

## Run

```
python techniques/ragas-rag-evaluation/python/ragas_rag_evaluation.py
Rscript techniques/ragas-rag-evaluation/r/ragas_rag_evaluation.R
```

**Refs:** Es, S., James, J., Espinosa-Anke, L. & Schockaert, S. "RAGAS: Automated evaluation of retrieval-augmented generation." *EACL*, 2024.

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
