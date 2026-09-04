# Latent Semantic Analysis / Indexing (Reference §42.13)

Deerwester et al. (1990), Landauer & Dumais (1997). Truncated SVD
of a term-document matrix `M` produces low-rank latent
representations of both terms and documents:

```
M ≈ U_k · Σ_k · V_k^T
```

- Rows of `U_k · Σ_k` = latent term vectors.
- Rows of `V_k^T · Σ_k` (i.e., `V_k · Σ_k`) = latent document
  vectors.

Documents that share **no literal terms** can still have high
cosine similarity in latent space — LSA captures **synonymy**.

## Query → latent space

`q̂ = q^T · U_k · Σ_k⁻¹`, then cosine-similarity to any document
vector.

## When to use

- **Information retrieval** on small-to-medium corpora before dense
  neural embeddings became standard.
- **Baseline** for topic-modelling-style questions.

## When NOT to use

- **Modern retrieval** with millions of documents — SBERT / BM25
  dominate.
- **Interpretability required** — LSA dimensions have no
  guaranteed semantic meaning.

## Files

- `python/lsa_latent_semantic.py` — truncated SVD of TDM + query
  projection + cosine ranking. Demo (6 short Deerwester-style
  documents, k=2): docs 0-3 (human-computer topic) all pairwise
  cos ≥ 0.83; docs 4-5 (graph-trees topic) separate; query "human
  computer interaction" ranks docs 0, 3, 1, 2 first (all HCI-
  adjacent).
- `r/lsa_latent_semantic.R` — `lsa::lsa`, `text2vec::LSA`,
  `irlba::irlba` (R); `sklearn.TruncatedSVD`, `gensim.LsiModel`
  (Python).

## Assumptions & caveats

- **Choice of `k`** — no principled way to pick; scree of singular
  values gives a hint.
- **Weighting** — TF-IDF or log-entropy weighting before SVD
  improves quality vs raw counts.
- **Static representation** — no context; the word "bank" gets one
  vector.
- **Sensitive to normalisation** — length-normalised documents
  behave better in cosine space.

## Related in this repo

- `tfidf-bm25`, `word-embeddings`, `topic-modeling-lda`,
  `document-embedding-similarity` — LSA cousins.
- `probabilistic-pca`, `nmf`, `sparse-pca` — matrix-factorisation
  alternatives.

## Run

```
python techniques/lsa-latent-semantic/python/lsa_latent_semantic.py
Rscript techniques/lsa-latent-semantic/r/lsa_latent_semantic.R
```

**Refs:** Deerwester, S., Dumais, S.T., Furnas, G.W., Landauer, T.K., & Harshman, R. "Indexing by latent semantic analysis." *JASIS*, 1990; Landauer, T.K. & Dumais, S.T. "A solution to Plato's problem: the latent semantic analysis theory." *Psychological Review*, 1997.

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
