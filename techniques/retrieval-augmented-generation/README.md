# Retrieval-Augmented Generation (RAG) (Reference §47.19)

Lewis et al. (2020). Combines a **parametric generator** (LM) with a
**non-parametric memory** (corpus + retriever). At inference:

1. **Encode** the user query as a dense vector `q`.
2. **Retrieve** top-`k` documents from the corpus by cosine
   similarity of `q` with document embeddings.
3. **Ground** the LM by concatenating retrieved passages with the
   query and generating conditioned on both.

## Advantages

- Injects **fresh / private** knowledge without model retraining.
- **Cites sources** (retrieved passages) — reduces hallucination.
- Smaller LM sufficient because facts come from the corpus, not
  parameters.

## Files

- `python/retrieval_augmented_generation.py` — minimal RAG pipeline:
  TF-IDF retriever + template-based "generator" from scratch. Demo
  (6-doc biostatistics corpus, query "survival model with
  censoring"): correctly retrieves Cox, RSF, and Kaplan-Meier docs
  and assembles a cited answer.
- `r/retrieval_augmented_generation.R` — `ellmer`, `chattr`,
  `quanteda`, `text2vec` (R); LangChain, LlamaIndex, sentence-
  transformers, transformers, vector DBs (chroma / weaviate / qdrant)
  (Python).

## When to use

- **Domain / private knowledge** — internal wikis, code, notes.
- **Frequently-updated facts** — swap the corpus without retraining.
- **Long-tail queries** — the corpus supplies rare facts a small LM
  can't memorise.
- **Grounded answers** — need source citations for audit / trust.

## When NOT to use

- **General reasoning tasks** — no external facts required; a
  well-tuned LM alone suffices.
- **Retrieval-hostile queries** — vague / conversational chatter with
  no lookup value.
- **Latency-critical inference** — retrieval adds a hop; consider
  distilling into the LM if speed dominates.

## Assumptions & caveats

- **Corpus quality** — garbage in, garbage cited. Curate.
- **Retriever recall** — miss the right passage and the LM
  hallucinates; use hybrid BM25 + dense retrieval and rerankers.
- **Context-window budget** — long retrievals blow context; use
  passage chunking + reranking.
- **Semantic drift** — the LM may ignore retrieved passages in favour
  of its priors ("Rag-Truth" faithfulness metrics).
- **Provenance / privacy** — retrievals may leak sensitive text;
  filter on access control.

## Related in this repo

- `document-embedding-similarity`, `tfidf-bm25`, `sentence-similarity`
  — the retrieval side.
- `text-generation-decoding`, `transformer-decoder`, `attention-mechanism`
  — the LM side.
- `question-answering` — extractive QA cousin.
- `knowledge-distillation` — compress an LM after it's proven
  useful via RAG.

## Run

```
python techniques/retrieval-augmented-generation/python/retrieval_augmented_generation.py
Rscript techniques/retrieval-augmented-generation/r/retrieval_augmented_generation.R
```

**Refs:** Lewis, P. et al. "Retrieval-augmented generation for knowledge-intensive NLP tasks." *NeurIPS*, 33: 9459-9474, 2020; Karpukhin, V. et al. "Dense passage retrieval for open-domain question answering." *EMNLP*, 2020.

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
