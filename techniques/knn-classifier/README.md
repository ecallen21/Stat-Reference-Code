# k-Nearest Neighbors (Reference §26.11)

Nonparametric prediction: for a new `x`, find the `k` closest training points and **majority-vote** (classification) or **average** (regression).

## Distance-weighted variant

Weight each neighbor by `1 / d` (or a Gaussian kernel). Reduces the impact of ties and puts more weight on close matches.

## Choosing k

- `k = 1` → perfect training accuracy, wildly overfitting.
- Large `k` → smoother, higher bias.
- **Cross-validate** on odd `k` in `[1, 30]`.

## Files

- `python/knn_classifier.py` — from-scratch kNN with distance-weighted option + CV-based k selection. Demo: k = 5 → 97% training accuracy; CV picks k = 11 with 3.3% CV error; matches sklearn KNeighborsClassifier.
- `r/knn_classifier.R` — `class::knn` or `FNN::knn`.

## When to use

- **Simple baseline** — no training cost, easy to explain.
- **Local nonlinear boundaries** where you don't want a parametric model.
- **Low-dimensional data** — kNN suffers the curse of dimensionality above `d ≈ 20`.
- **DTW-based time-series classification** (see `ts-features-classification`).

## When NOT to use

- **High-dimensional data** — distances become meaningless.
- **Large training sets** (n > 10⁵) — prediction cost `O(n)` per query without indexing.
- **Very imbalanced classes** — majority class dominates neighbors.

## Assumptions & caveats

- **Standardize features** — Euclidean distance is scale-sensitive.
- **Curse of dimensionality** — use PCA / kernel PCA / UMAP to reduce dimension first.
- **Ball-tree / KD-tree indexing** — `O(log n)` per query for low `d`.
- **Approximate nearest neighbors** (LSH, HNSW) for very large scale.

## Run

```
python techniques/knn-classifier/python/knn_classifier.py
Rscript techniques/knn-classifier/r/knn_classifier.R
```

**Refs:** Cover, T. & Hart, P. "Nearest neighbor pattern classification." *IEEE Trans. Inf. Theory* 13(1), 21–27, 1967; Hastie, T., Tibshirani, R. & Friedman, J. *The Elements of Statistical Learning*, 2nd ed., Springer, 2009 (Ch 2.3, 13.3).

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
