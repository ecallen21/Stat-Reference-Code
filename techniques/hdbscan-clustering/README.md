# HDBSCAN Clustering (Reference §47.79)

Campello, Moulavi & Sander (2013). Hierarchical DBSCAN — removes
DBSCAN's `eps` parameter. Steps:

    1. Mutual-reachability distance d_m(a, b) = max(core(a),
       core(b), dist(a, b)).
    2. MST of d_m.
    3. Single-linkage dendrogram.
    4. Extract flat clusters by MAXIMUM PERSISTENCE.

Handles clusters of VARYING DENSITY, marks low-density outliers
as noise (label `= −1`). Only parameter: `min_cluster_size`.

## Files

- `python/hdbscan_clustering.py` — wrapper over
  `sklearn.cluster.HDBSCAN` (or the standalone `hdbscan`
  package). Demo (3 clusters of dramatically different density
  (σ=0.15, 0.6, 1.2) plus uniform noise):
  - HDBSCAN at min_cluster_size=15 → 3 clusters, 75 noise pts,
    ARI 0.775.
  - K-means k=3 baseline → ARI 0.743 but absorbs noise.
- `r/hdbscan_clustering.R` — `dbscan::hdbscan` (R);
  `hdbscan`, `sklearn.cluster.HDBSCAN`, from-scratch reference
  in Python.

## When to use

- **Clusters of varying density** — HDBSCAN's key advantage over
  DBSCAN.
- **Noise-heavy data** — the −1 label separates outliers cleanly.
- **Unknown K** — HDBSCAN discovers cluster count from persistence.
- **Non-convex clusters** in low-to-mid dim.

## When NOT to use

- **Very high-dim data** — distance concentration hurts density
  estimation; reduce dim first.
- **Streaming data** — batch-only; use incremental variants.
- **When K is known and clusters are convex + equal-density** —
  k-means is faster and simpler.

## Assumptions & caveats

- **min_cluster_size** trades granularity vs noise absorption.
- **min_samples** (default = min_cluster_size) controls core-point
  density.
- **Distance metric** matters; use Mahalanobis or precomputed
  distances when Euclidean is inappropriate.
- **Cluster persistence** favours long-lived clusters; some
  meaningful small clusters may be missed.

## Related in this repo

- `dbscan` — the ε-based ancestor.
- `k-means`, `k-medoids`, `gaussian-mixture-models`,
  `hierarchical-clustering`, `spectral-clustering`,
  `affinity-propagation` — clustering peers.
- `hopkins-clusterability`, `silhouette-clusters`,
  `davies-bouldin-index`, `gap-statistic-cluster` — validation.
- `isolation-forest-anomaly` — noise / outlier detection cousin.

## Run

```
python techniques/hdbscan-clustering/python/hdbscan_clustering.py
Rscript techniques/hdbscan-clustering/r/hdbscan_clustering.R
```

**Refs:** Campello, R.J.G.B., Moulavi, D. & Sander, J. "Density-based clustering based on hierarchical density estimates." *PAKDD*, 2013; McInnes, L., Healy, J. & Astels, S. "hdbscan: Hierarchical density based clustering." *JOSS* 2(11): 205, 2017.

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
