# LODA Anomaly Detection (Reference §47.135)

Pevný (2016). Anomaly detection via a collection of RANDOM 1-D
projections plus per-projection histogram density estimates:

    score(x) = − (1/K) Σₖ log p_k( w_kᵀ x ),  higher = more anomalous.

Sparse Gaussian weights `w_k` keep memory low. Streaming-friendly
(histograms update incrementally), and competitive with
IsolationForest / One-Class SVM on many benchmarks.

## Files

- `python/loda_anomaly_detection.py` — from-scratch LODA with
  sparse random projections + numpy histograms. Demo (500 normal
  + 20 anomalous points in 20-D, K=100 projections):
  - **LODA top-20 precision = 1.000**
  - IsolationForest top-20 precision = 1.000.
- `r/loda_anomaly_detection.R` — no first-class R port;
  `pyod.models.LODA` (Python).

## When to use

- **Streaming anomaly detection** — histograms + random projections
  update online.
- **High-dim tabular** — sparse projections keep memory tame.
- **Low compute budget** — one of the cheapest AD algorithms.
- **Ensemble anomaly voting** — combine with IF / OCSVM.

## When NOT to use

- **Structured anomalies in low-d** — density-based methods (KDE,
  DBSCAN, HDBSCAN) may be sharper.
- **When explanations matter** — projection weights are random,
  interpretation is limited.
- **Very small n** — histograms unstable; use kNN or KDE instead.

## Assumptions & caveats

- **Sparsity** in projection weights — √d nonzeros per projection
  is a common default.
- **# histograms bins** — 10-30 typical; more bins fit tighter but
  noisier.
- **K projections** — larger K → smoother score.
- **Contamination assumption** — training set assumed to be mostly
  normal.

## Related in this repo

- `isolation-forest-anomaly`, `one-class-svm`,
  `mahalanobis-distance` (via `multivariate-outlier-detection`) —
  anomaly-detection cousins.
- `random-projections`, `random-fourier-features`,
  `product-quantization-pq`, `min-hash-lsh` — random-sketch
  neighbours.
- `hopkins-clusterability`, `silhouette-clusters`,
  `davies-bouldin-index` — cluster-quality diagnostics.
- `distance-correlation`, `hoeffding-d-independence` — related
  distributional-comparison ideas.

## Run

```
python techniques/loda-anomaly-detection/python/loda_anomaly_detection.py
Rscript techniques/loda-anomaly-detection/r/loda_anomaly_detection.R
```

**Refs:** Pevný, T. "Loda: Lightweight on-line detector of anomalies." *Machine Learning* 102(2): 275-304, 2016.

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
