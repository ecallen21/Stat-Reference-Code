# TF-IDF and BM25 (Reference §25.2)

Two workhorse term-weighting schemes for classical information retrieval and
text mining.

## TF-IDF

Term frequency × inverse document frequency:

```
tf(t, d)   = raw or log-scaled count of term t in doc d
idf(t)     = log((N + 1) / (df(t) + 1)) + 1       (sklearn smoothing)
tf-idf     = tf(t, d) · idf(t)   (usually L2-normalised per doc)
```

- Down-weights common terms; up-weights document-discriminative terms.
- Widely used as input features for classifiers, similarity search, topic models.

## Okapi BM25

Best-Match 25 (Robertson-Sparck Jones et al. 1994) — the workhorse ranking
function for search engines pre-neural (and still competitive today):

```
BM25(t, d) = idf_bm25(t) · tf(t, d) · (k1 + 1)
             / [ tf(t, d) + k1 · (1 − b + b · |d| / avg|d|) ]

idf_bm25(t) = log( (N − df(t) + 0.5) / (df(t) + 0.5) + 1 )      (clip at 0)
```

- `k1 ∈ [1.2, 2.0]` — term-frequency saturation (BM25 stops rewarding repeated terms past this).
- `b ∈ [0, 1]` — length normalisation (0 = none, 1 = full).
- Together they beat plain TF-IDF for ad-hoc retrieval by ~10–20% MAP on TREC.

## When to use

- **TF-IDF** — features for classical text classification / clustering / topic modelling; interpretable weights.
- **BM25** — the default retrieval scoring for classical search (Elasticsearch, Lucene, Whoosh).
- **Hybrid** — combine BM25 with dense-embedding retrieval (e.g. `sentence-transformers`) for the "sparse + dense" retrieval pipelines now common.

## Files

- `python/tfidf_bm25.py` — from-scratch TF-IDF vectoriser + BM25 scorer. Demo (5 pangram-style docs, query "quick brown fox"): IDF of "the" = 1.41 (common), IDF of "stitch" = 2.10 (rare); doc L2 norm = 1.000; BM25 ranks doc 4 (repeats query terms) above doc 0 (single occurrence). sklearn cross-check matches TF-IDF norm.
- `r/tfidf_bm25.R` — `text2vec::TfIdf$new()`, `quanteda::dfm_tfidf`, `tm::weightTfIdf`, `superml::TfIdfVectorizer`.

## Assumptions & caveats

- **Bag-of-words** — order and syntax discarded. Use bigrams / `ngram_range` if phrase discrimination matters.
- **Vocabulary explosion** — n-grams inflate feature space quadratically; prune by `min_df` / `max_df` or hash the vocabulary (`HashingVectorizer`).
- **Sparse but wide** matrices — always keep in scipy sparse format; dense representations blow memory.
- **Cross-corpus IDF drift** — IDF fitted on training data may be poorly calibrated for a new corpus; refit or use pretrained embeddings for out-of-distribution documents.
- **BM25's saturation ≠ TF-IDF's linear rise** — BM25 is better when document lengths vary widely.
- **Query expansion / rewriting** (RM3, doc2query) further improves both — orthogonal to the scoring choice.

## Related in this repo

- `text-preprocessing` — the tokenisation / normalisation input pipeline.
- `document-clustering`, `text-classification`, `topic-modeling-lda` — TF-IDF-based downstream models.
- `word-embeddings` — dense alternative to sparse TF-IDF.

## Run

```
python techniques/tfidf-bm25/python/tfidf_bm25.py
Rscript techniques/tfidf-bm25/r/tfidf_bm25.R
```

**Refs:** Salton, G. & Buckley, C. "Term-weighting approaches in automatic text retrieval." *Inf. Process. Manag.* 24(5), 513–523, 1988; Robertson, S.E. et al. "Okapi at TREC-3." *NIST Special Publication* 500-225, 1995; Manning, C.D., Raghavan, P. & Schütze, H. *Introduction to Information Retrieval*, Cambridge UP, 2008.

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
