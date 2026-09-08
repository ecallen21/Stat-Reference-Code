# Hopkins Clusterability Statistic (Reference §47.73)

Hopkins & Skellam (1954). Test whether a dataset has meaningful
**cluster structure** before running any clustering algorithm.

    1. Sample m Y-points uniformly from the bounding box of X.
    2. Sample m X-points from the dataset.
    3. u_i = dist(Y_i, nearest X-point).
    4. w_i = dist(X_i, nearest OTHER X-point).
    5. H = Σ u_i / (Σ u_i + Σ w_i)   ∈ [0, 1].

Interpretation:
  - H ≈ 0.5: Poisson-like, no cluster structure.
  - H > 0.75: highly clustered.
  - H < 0.25: regularly spaced.

## Files

- `python/hopkins_clusterability.py` — from-scratch Hopkins
  with `scipy.spatial.cKDTree`. Demo (averages over 20 seeds):
  - Uniform on [0, 1]³ (n=500)                   H = 0.502
  - 3-cluster Gaussian mixture (n=450)           H = 0.970
  - Regular 2-D grid (n=400)                     H = 0.274.
- `r/hopkins_clusterability.R` — `clustertend::hopkins`,
  `factoextra::get_clust_tendency` (R); `pyclustertend`,
  `hopkins`, from-scratch (Python).

## When to use

- **Cluster-tendency diagnostic** — before k-means / hierarchical
  / DBSCAN.
- **Companion to gap statistic / silhouette** — first ask "is
  there any structure?" then "how many clusters?".
- **Sanity check on embeddings** — post-UMAP / t-SNE.

## When NOT to use

- **Very small n** — H unstable; use multiple bootstrap seeds.
- **Highly correlated dimensions** — bounding-box sampling is
  biased; whiten first.
- **Categorical data** — need Gower distance and separate
  clusterability definition.

## Assumptions & caveats

- **Randomness** — average H over multiple seeds; single-shot can
  mislead.
- **Bounding-box sampling** biased for non-hyperrectangular support
  — sample from convex hull or empirical distribution instead.
- **Dimensionality curse** — Hopkins loses power in very high d;
  use dimension-reduction first.
- **Not a null-hypothesis test** — a threshold, not a p-value.

## Related in this repo

- `silhouette-clusters` — companion cluster-validation index.
- `k-means`, `k-medoids`, `gaussian-mixture-models`,
  `hierarchical-clustering`, `dirichlet-process-mixture` —
  clustering algorithms Hopkins pre-screens for.
- `tsne-umap`, `isomap`, `lle-locally-linear-embedding`,
  `diffusion-maps` — embeddings that may reveal / hide structure.
- `ripleys-k-point-pattern`, `spatial-scan-cluster` — spatial
  point-pattern analogues.

## Run

```
python techniques/hopkins-clusterability/python/hopkins_clusterability.py
Rscript techniques/hopkins-clusterability/r/hopkins_clusterability.R
```

**Refs:** Hopkins, B. & Skellam, J.G. "A new method for determining the type of distribution of plant individuals." *Annals of Botany* 18(2): 213-227, 1954; Lawson, R.G. & Jurs, P.C. "New index for clustering tendency and its application to chemical problems." *J Chem Inf Comput Sci* 30(1): 36-41, 1990.

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
