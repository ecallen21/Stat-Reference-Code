# Functional Clustering (Reference §31.6)

Group curves `x_i(t)` into K clusters.

## Two practical routes

- **FPC + k-means** — top-K functional PC scores fed to k-means.
- **Basis-coefficient clustering** — spline / Fourier coefficient
  vectors as cluster features.

Model-based alternatives (James-Sugar 2003 funHDDC, Bouveyron-Jacques
2011) fit mixtures of functional distributions.

## When to use

- **Time-series subtyping** — patient-day glucose curves, EEG epochs,
  sensor day-profiles.
- **Shape-based grouping** — a natural discovery task on curve
  datasets.
- **Downstream cohort analysis** on the recovered clusters.

## When NOT to use

- **Phase misalignment dominates** — warp first via
  `curve-registration`.
- **Very sparse / irregular** — use PACE-based clustering.
- **Interpretability** of clusters matters and shapes overlap —
  model-based clustering with soft assignments better.

## Files

- `python/functional_clustering.py` — from-scratch FPCA scores +
  k-means with 10-restart initialisation and `cluster_purity`.
  Demo: three curve families (sin, damped-sin, ramp): **cluster
  purity 1.000** with K = 3 clusters and 3 FPCs.
- `r/functional_clustering.R` — `funHDDC`, `fda.usc::kmeans.fd`,
  `Funclustering`, `fdapace` (R); `scikit-fda`, `tslearn` (Python).

## Assumptions & caveats

- **K selection** — silhouette, BIC (model-based), or domain-priors.
- **Distance metric** — L² is default; DTW for time-warped curves;
  amplitude / phase distances for shape.
- **Init sensitivity** — always use multiple restarts.
- **Model-based vs partitional** — model-based gives soft memberships
  and BIC-driven K.
- **Alignment** — do phase alignment before clustering if warping is
  large.

## Related in this repo

- `functional-pca` — the score generator.
- `functional-regression`, `functional-anova`, `curve-registration`,
  `functional-depth` — FDA family (this batch).
- `k-means`, `hierarchical-clustering`, `dbscan` — multivariate
  clustering cousins.
- `dynamic-time-warping` (if present) — warp-aware distance.

## Run

```
python techniques/functional-clustering/python/functional_clustering.py
Rscript techniques/functional-clustering/r/functional_clustering.R
```

**Refs:** James, G.M. & Sugar, C.A. "Clustering for sparsely sampled functional data." *JASA*, 2003; Bouveyron, C. & Jacques, J. "Model-based clustering of time series in group-specific functional subspaces." *Advances in Data Analysis and Classification*, 2011.

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
