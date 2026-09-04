# Document Embedding + Similarity (Reference §42.9)

Reimers & Gurevych (2019), Robertson-Sparck-Jones (1976). Represent
a document as a vector, then rank / cluster / de-duplicate by
similarity.

## Options

- **TF-IDF + cosine** — sparse baseline; strong when vocabulary
  overlap matters.
- **Doc2Vec / SBERT** — dense neural embeddings; capture semantic
  similarity even without lexical overlap.
- **BM25** — classical IR ranking, uses length normalisation +
  IDF (`k1`, `b` are the standard hyperparameters).

## When to use

- **Retrieval** — return top-k docs matching a query.
- **Duplicate detection** — clinical notes, PR-comment
  deduplication.
- **Clustering / recommendation** — group similar documents.

## When NOT to use

- **Very short queries** with rare vocabulary — TF-IDF baseline
  underperforms neural models.
- **Cross-lingual** — plain TF-IDF fails; use multilingual SBERT.

## Files

- `python/document_embedding_similarity.py` — TF-IDF + cosine
  similarity matrix + BM25 ranker (custom). Demo (4 clinical /
  sports notes, query "aspirin chest pain"): BM25 ranks
  **doc 0 (2.14) > doc 1 (2.02) > sport / diabetes docs (0)**;
  TF-IDF cosine picks up 0.54 similarity between the two aspirin
  notes.
- `r/document_embedding_similarity.R` — `text2vec::doc2vec`,
  `text (sentence-transformers)`, `quanteda` (R);
  `sentence-transformers`, `gensim::Doc2Vec`, `sklearn.
  TfidfVectorizer`, `rank_bm25` (Python).

## Assumptions & caveats

- **Length normalisation** — cosine on raw counts privileges long
  docs; TF-IDF or length-normalised BM25 handles this.
- **Stop-word removal** hurts retrieval when the query term is a
  stop word ("no" in clinical negation).
- **BM25 hyperparameters** — `k1 ≈ 1.5`, `b ≈ 0.75` are common
  defaults; tune per corpus.
- **Neural vs sparse** — SBERT dominates for paraphrase, TF-IDF
  wins for exact-match retrieval.

## Related in this repo

- `tfidf-bm25`, `word-embeddings`, `topic-modeling-lda` — companion
  representations.
- `named-entity-recognition`, `relation-extraction` — feed downstream
  IR pipelines.

## Run

```
python techniques/document-embedding-similarity/python/document_embedding_similarity.py
Rscript techniques/document-embedding-similarity/r/document_embedding_similarity.R
```

**Refs:** Reimers, N. & Gurevych, I. "Sentence-BERT: sentence embeddings using Siamese BERT-networks." *EMNLP*, 2019; Robertson, S. & Sparck Jones, K. "Relevance weighting of search terms." *JASIS*, 1976.

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
