# Isolation Forest + One-Class SVM + Elliptic Envelope (Reference §26.18)

**Anomaly detection**: find rare, "different" observations without a labeled `y`. Contrast with time-series anomaly detection (`ts-anomaly-detection`), which uses temporal structure.

## Isolation Forest (Liu-Ting-Zhou 2008)

Build many random trees. At each node pick a random feature and random threshold, recurse until each point is isolated. **Anomalies get isolated in shorter path length** → anomaly score:

```
s(x) = 2^(−E[h(x)] / c(n))
```

- `s ≈ 0.5` — normal (expected path length)
- `s → 1` — anomalous (very short path)

## One-Class SVM (Schölkopf 2001)

Learn a decision boundary that encloses the majority "normal" data; observations outside are anomalies. RBF kernel default.

## Elliptic Envelope (Rousseeuw 1999)

Fit robust covariance (MCD) to normal data (see `multivariate-outlier-detection`). Mahalanobis distance from the robust centre is the anomaly score. Works well when normal data is roughly Gaussian; poor for multimodal.

## Files

- `python/isolation_forest_anomaly.py` — from-scratch Isolation Forest with random-tree path-length scoring; sklearn cross-check. Demo (200 normal + 10 obvious outliers in 3-D): precision 0.82, recall 0.90 — matches `sklearn.ensemble.IsolationForest` exactly.
- `r/isolation_forest_anomaly.R` — `isotree::isolation.forest` or `solitude::isolationForest`.

## When to use each

- **Isolation Forest** — default first choice; fast, no tuning, works with mixed feature types.
- **One-Class SVM** — good for boundary-like normal regions; needs `nu` tuning.
- **Elliptic Envelope** — best when normal data is Gaussian and low-dimensional; fastest.
- **Autoencoder / VAE reconstruction error** — deep-learning alternative for images / high-D data.

## Related methods

- **Local Outlier Factor (LOF)** — density-based; catches local anomalies inside clusters.
- **DBSCAN** noise points — clustering-based (see `dbscan`).
- **Robust Mahalanobis (MCD)** (see `multivariate-outlier-detection`) — same idea as Elliptic Envelope.

## Assumptions & caveats

- **Contamination** — the estimated fraction of anomalies matters for setting the threshold; use domain knowledge or ROC on validation labels if available.
- **High dimensions** — Isolation Forest is more robust than density methods, but consider dimension reduction first.
- **Unlabeled evaluation** — hardest part; if you have any labels, evaluate precision-at-K on the flagged set.

## Run

```
python techniques/isolation-forest-anomaly/python/isolation_forest_anomaly.py
Rscript techniques/isolation-forest-anomaly/r/isolation_forest_anomaly.R
```

**Refs:** Liu, F.T., Ting, K.M. & Zhou, Z.-H. "Isolation forest." *ICDM*, 2008; Schölkopf, B. et al. "Estimating the support of a high-dimensional distribution." *Neural Comput.* 13(7), 1443–1471, 2001; Rousseeuw, P.J. & Van Driessen, K. "A fast algorithm for the minimum covariance determinant estimator." *Technometrics* 41(3), 212–223, 1999.

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
