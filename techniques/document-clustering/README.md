# Document Clustering (Reference §25.5)

Cluster documents represented as TF-IDF vectors using **spherical k-means**
(equivalent to k-means on L2-normalised vectors, which is cosine similarity).

## Pipeline

1. Preprocess text (`text-preprocessing`).
2. TF-IDF vectorise (`tfidf-bm25`), L2-normalise per row.
3. Spherical k-means with several random restarts, choose the best by cosine cohesion.
4. Evaluate against known labels (when available): purity, NMI, ARI.

## Evaluation metrics

| Metric | Range | Meaning |
|---|---|---|
| **Purity** | [0, 1] | Fraction of docs in the dominant true class per cluster (biased toward many small clusters). |
| **NMI** | [0, 1] | Mutual information / mean entropy. Chance-corrected variants exist. |
| **ARI** | [−1, 1] | Rand index, adjusted for chance. 0 = chance, 1 = perfect. |
| **V-measure** | [0, 1] | Harmonic mean of homogeneity + completeness. |

## When to use

- **Exploratory analysis** — group similar documents without labels.
- **Deduplication / near-duplicate detection**.
- **Feature reduction** — cluster IDs as coarse features for a downstream classifier.
- **Search-result diversification** — cluster then present one per cluster.

## Files

- `python/document_clustering.py` — from-scratch spherical k-means with `n_starts` restarts + purity / NMI / ARI. Demo (D=60 docs from 3 disjoint topical vocabularies of 10 words each, K=3): purity 1.000, NMI 1.000, ARI 1.000; sklearn `KMeans` on the same TF-IDF matches exactly.
- `r/document_clustering.R` — `text2vec` + `skmeans::skmeans`, `cluster::pam`, `aricode::NMI / ARI`.

## Related methods

- **k-medoids / PAM** — robust to outliers; use for skewed clusters.
- **DBSCAN / HDBSCAN** — non-convex clusters, arbitrary shapes; require good distance metric on high-dim TF-IDF.
- **Latent Dirichlet Allocation** — soft-clustering probabilistic alternative (see `topic-modeling-lda`).
- **BERTopic / c-TF-IDF** — modern transformer-embedding + HDBSCAN pipeline; often better on real corpora.
- **Hierarchical clustering** — for a dendrogram / interactive drill-down.

## Assumptions & caveats

- **Sparse but wide** — always use sparse matrix representations; dense TF-IDF blows memory past ~10k docs × 20k vocab.
- **Cosine vs Euclidean** — TF-IDF is L2-normalised so k-means on it *is* spherical k-means; use spherical explicitly for clarity.
- **k unknown** — silhouette, elbow, gap statistic on TF-IDF is noisy; try 3–5 K values and inspect cluster words.
- **Class imbalance** — very rare clusters get missed; use k-medoids or over-sampled initialisation.
- **Multiple restarts are essential** — random-init spherical k-means gets stuck in bad optima (the demo shows a single-run purity of 0.67 vs 1.00 after restarts).
- **Interpretability** — report top-TF-IDF terms per cluster (like LDA topic words).

## Related in this repo

- `text-preprocessing`, `tfidf-bm25`, `word-embeddings` — input representations.
- `k-means`, `k-medoids`, `dbscan`, `gaussian-mixture-models` — general clustering.
- `cluster-validation` — evaluation metrics for clustering.
- `topic-modeling-lda` — soft-clustering alternative.

## Run

```
python techniques/document-clustering/python/document_clustering.py
Rscript techniques/document-clustering/r/document_clustering.R
```

**Refs:** Steinbach, M., Karypis, G. & Kumar, V. "A comparison of document clustering techniques." *KDD Workshop on Text Mining*, 2000; Dhillon, I.S. & Modha, D.S. "Concept decompositions for large sparse text data using clustering." *Machine Learning* 42(1-2), 143–175, 2001.

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
