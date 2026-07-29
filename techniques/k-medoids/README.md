# k-medoids / PAM — Partitioning Around Medoids (Reference §9.10)

Like `k-means`, but:

- **Centers are actual data points** (medoids), not means.
- Works with **any distance metric** — Manhattan, Gower for mixed data, precomputed dissimilarities, etc.
- **More robust to outliers** than k-means.

## PAM algorithm (Kaufman & Rousseeuw 1987)

- **BUILD phase**: greedy — add one medoid at a time, each chosen to minimize total cost given the ones already selected.
- **SWAP phase**: try swapping each medoid with each non-medoid; accept any swap that reduces total cost. Repeat until no improvement.

Cost = `Σ_i d(x_i, medoid(cluster(x_i)))`.

## Trade-off

- Complexity `O(K(n − K)²)` per iteration — slower than k-means' `O(nK)`.
- For large `n` use **CLARA** (Kaufman & Rousseeuw) or **CLARANS** (Ng & Han) — sample-based approximations.

## Files

- `python/k_medoids.py` — from-scratch PAM with BUILD + SWAP phases; supports Euclidean and Manhattan; optional cross-check `sklearn_extra.cluster.KMedoids`.
- `r/k_medoids.R` — thin wrapper around `cluster::pam` (authoritative).

## Assumptions

- Precomputed or computable pairwise distances.
- `k` chosen a priori (or via `cluster-validation` methods).

## Run

```
python techniques/k-medoids/python/k_medoids.py
Rscript techniques/k-medoids/r/k_medoids.R
```

**Refs:** Kaufman, L. & Rousseeuw, P.J. "Clustering by means of medoids." In *Statistical Data Analysis Based on the L1-Norm*, 405–416, North-Holland, 1987; Kaufman, L. & Rousseeuw, P.J. *Finding Groups in Data*, Wiley, 1990.

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
