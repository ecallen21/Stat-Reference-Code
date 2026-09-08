# Silhouette Clustering Validation (Reference §47.74)

Rousseeuw (1987). Per-point silhouette:

    a_i = mean distance to other points in its cluster
    b_i = min over other clusters of mean distance to them
    s_i = (b_i − a_i) / max(a_i, b_i)   ∈ [−1, 1].

Mean silhouette width = 1/n Σ s_i is a widely used cluster-quality
score. Rousseeuw's rule of thumb:
  - > 0.7: strong structure
  - 0.5–0.7: reasonable
  - 0.25–0.5: weak, may be artefactual
  - < 0.25: probably no substantial structure.

## Files

- `python/silhouette_clusters.py` — from-scratch per-point +
  mean silhouette with `sklearn.KMeans` fits. Demo (n=300, 3
  true clusters):
  - well-separated: k=2 → 0.66, k=3 → **0.82**, k=4 → 0.66
  - overlapping:   k=2 → 0.38, k=3 → **0.45**, k=4 → 0.40
  In both cases the max-silhouette choice matches the truth K=3.
- `r/silhouette_clusters.R` — `cluster::silhouette`,
  `factoextra::fviz_silhouette` (R); `sklearn.metrics.
  silhouette_score`, from-scratch (Python).

## When to use

- **Choosing K** in k-means / k-medoids / GMM / hierarchical
  clustering.
- **Detecting misassigned points** — per-point negative s_i signals
  a probable wrong-cluster assignment.
- **Comparing clustering algorithms** at fixed K.
- **Post-hoc validation** of DBSCAN / OPTICS results.

## When NOT to use

- **Very large n** — O(n²) memory / compute; subsample or use
  approximate silhouette (Sil-K-means).
- **Density-based clusters with noise** — DBSCAN's noise class
  distorts a_i; use Davies-Bouldin or CH-index instead.
- **Highly imbalanced clusters** — a_i term dominated by the big
  cluster.

## Assumptions & caveats

- **Distance choice matters** — Euclidean vs Manhattan vs cosine.
- **Standardise features** first when scales differ.
- **Non-metric distances** — silhouette not defined; use non-
  metric alternatives (medoid-based).
- **Reports one number** — pair with the SILHOUETTE PLOT
  (per-cluster distributions) for diagnosis.

## Related in this repo

- `hopkins-clusterability` — companion "is there structure at all?"
  test.
- `k-means`, `k-medoids`, `gaussian-mixture-models`,
  `hierarchical-clustering`, `dirichlet-process-mixture`,
  `latent-profile-analysis` — clustering algorithms to score.
- `functional-clustering` — silhouette works on curves too.
- `wgcna-coexpression`, `latent-space-network` — module-detection
  cousins.

## Run

```
python techniques/silhouette-clusters/python/silhouette_clusters.py
Rscript techniques/silhouette-clusters/r/silhouette_clusters.R
```

**Refs:** Rousseeuw, P.J. "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis." *J Comp Appl Math* 20: 53-65, 1987; Kaufman, L. & Rousseeuw, P.J. *Finding Groups in Data.* Wiley, 1990.

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
