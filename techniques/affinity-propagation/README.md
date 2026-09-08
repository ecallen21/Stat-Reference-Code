# Affinity Propagation (Reference §47.80)

Frey & Dueck (2007). Message-passing algorithm that
simultaneously identifies EXEMPLARS and cluster memberships from a
similarity matrix. Number of clusters is set implicitly by the
DIAGONAL "preference" — median-of-similarities is a common
default. Alternates:

    responsibility  r(i, k) = s(i, k) − max_{k' ≠ k} (a(i, k') + s(i, k'))
    availability    a(i, k) = min(0, r(k, k) + Σ_{i' ≠ i, k} max(0, r(i', k))).

Damped by λ ∈ [0.5, 1). Point i's exemplar = argmax_k (a(i, k) + r(i, k)).

## Files

- `python/affinity_propagation.py` — from-scratch AP loop
  (Frey-Dueck update rules) plus a sklearn `AffinityPropagation`
  demo. Demo (4-corner Gaussian blobs, n=200):
  - preference q05 (−81.1) → 4 clusters, ARI 1.000
  - preference q25 (−53.6) → 4 clusters, ARI 1.000
  - preference q50 (−35.9) → 4 clusters, ARI 1.000
  - preference q75 (−17.5) → 4 clusters, ARI 1.000.
- `r/affinity_propagation.R` — `apcluster::apcluster` (R);
  `sklearn.cluster.AffinityPropagation`, from-scratch (Python).

## When to use

- **Unknown K**, plus you want each cluster represented by a REAL
  data point (exemplar) rather than a centroid.
- **Non-Euclidean similarities** — any negative-distance / kernel
  matrix works.
- **Bioinformatics / gene-cluster discovery** — Frey-Dueck original
  application.
- **Prototype selection** for downstream nearest-prototype classifiers.

## When NOT to use

- **Very large n** — O(n²) memory / update; sparse-AP or hierarchical
  variants for scale.
- **When K is known** — k-means or GMM often faster / more stable.
- **When exemplars aren't needed** — spectral or hierarchical give
  equally good partitions with less compute.

## Assumptions & caveats

- **Preference tuning** — controls K implicitly; sweep and pick by
  silhouette / gap / BIC.
- **Damping** typically 0.5–0.9; too low → oscillation.
- **Convergence** monitored by stability of exemplar set for
  ≥ 10 consecutive iterations.
- **Not deterministic**: with equal similarities you can get
  different exemplars across runs.

## Related in this repo

- `k-means`, `k-medoids`, `gaussian-mixture-models`,
  `hierarchical-clustering`, `spectral-clustering`,
  `hdbscan-clustering`, `dbscan` — clustering peers.
- `hopkins-clusterability`, `silhouette-clusters`,
  `davies-bouldin-index`, `gap-statistic-cluster` — validation.
- `patient-similarity-network`, `latent-space-network` — graph /
  similarity cousins.
- `plackett-luce-ranking`, `bradley-terry` — competing message-
  passing / probabilistic-choice ideas.

## Run

```
python techniques/affinity-propagation/python/affinity_propagation.py
Rscript techniques/affinity-propagation/r/affinity_propagation.R
```

**Refs:** Frey, B.J. & Dueck, D. "Clustering by passing messages between data points." *Science* 315(5814): 972-976, 2007.

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
