# Davies-Bouldin Cluster Index (Reference §47.82)

Davies & Bouldin (1979). Ratio of intra-cluster scatter to
inter-cluster distance:

    S_i = mean_{x ∈ C_i} ‖x − centroid_i‖
    M_ij = ‖centroid_i − centroid_j‖
    R_ij = (S_i + S_j) / M_ij
    DB = (1/K) Σ_i max_{j ≠ i} R_ij.

**LOWER = better** (0 for perfectly compact, well-separated
clusters). Choose K by minimising DB.

## Files

- `python/davies_bouldin_index.py` — from-scratch DB
  computation. Demo:
  - Well-separated 3-cluster data: k=2 → 0.66, **k=3 → 0.16**,
    k=4 → 0.67 (min at truth).
  - Overlapping 3-cluster data:    k=2 → 0.95, **k=3 → 0.59**,
    k=4 → 0.81 (min still at truth).
- `r/davies_bouldin_index.R` — `fpc::cluster.stats`,
  `clusterCrit::intCriteria` (R);
  `sklearn.metrics.davies_bouldin_score`, from-scratch (Python).

## When to use

- **Cheap alternative to silhouette** — O(K²) rather than O(n²)
  after clusters known.
- **Selecting K** alongside silhouette / gap for a triangulated
  choice.
- **Grid search** over clustering hyperparameters.
- **Model comparison** at fixed K across algorithms.

## When NOT to use

- **Non-convex clusters** — DB assumes centroids are meaningful;
  fails on rings / moons.
- **Density-based clusters with noise** — noise inflates scatter
  S_i; use DBCV or silhouette (excluding noise) instead.
- **Very unequal cluster sizes** — small clusters dominate the max
  ratio; complement with per-cluster diagnostics.

## Assumptions & caveats

- **Convex, roughly-Gaussian clusters** — DB implicitly assumes
  centroids summarise clusters.
- **Distance choice** — Euclidean standard; Manhattan / cosine
  variants exist.
- **Not a monotonic function of K** — larger K may temporarily
  raise DB before it drops again; scan a range.
- **Sensitive to outliers** — Winsorise or use trimmed
  centroids for robustness.

## Related in this repo

- `silhouette-clusters`, `gap-statistic-cluster`,
  `hopkins-clusterability` — companion cluster diagnostics.
- `k-means`, `k-medoids`, `gaussian-mixture-models`,
  `hierarchical-clustering`, `spectral-clustering`,
  `hdbscan-clustering`, `affinity-propagation` — clustering peers.
- `information-criteria` — BIC / AIC for probabilistic mixtures.
- `functional-clustering` — DB extends to curve data with L2
  functional distance.

## Run

```
python techniques/davies-bouldin-index/python/davies_bouldin_index.py
Rscript techniques/davies-bouldin-index/r/davies_bouldin_index.R
```

**Refs:** Davies, D.L. & Bouldin, D.W. "A cluster separation measure." *IEEE TPAMI* PAMI-1(2): 224-227, 1979.

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
