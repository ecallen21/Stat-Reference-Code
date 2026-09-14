# MCD — Minimum Covariance Determinant (Reference §47.374)

Rousseeuw (1985); Rousseeuw-Van Driessen (1999, FAST-MCD).
Multivariate cousin of LTS. Find the h-subset out of `n` with
the SMALLEST covariance determinant. The resulting location
vector `T` and covariance matrix `C` are ROBUST with 50%
breakdown point and are Fisher-consistent at multivariate
Gaussian after a scale correction.

## Files

- `python/mcd_robust_covariance.py` — n=200, p=3, 30 % wild
  uniform-hypercube outliers. Sample-mean error MLE 2.27 vs
  MCD 0.32 (7× tighter); sample-cov Frobenius relative error
  MLE 6.75 vs MCD 1.12 (6× tighter). True mean `(1, 2, 3)`,
  MCD estimate `(1.30, 1.98, 3.12)`.
- `r/mcd_robust_covariance.R` — `robustbase::covMcd` (the
  Rousseeuw reference), `rrcov::CovMcd` (R);
  `sklearn.covariance.MinCovDet`, `EllipticEnvelope`,
  from-scratch FAST-MCD (Python).

## When to use

- **Robust multivariate location / scale** — Mahalanobis
  outlier flagging, robust PCA start, MDS distance.
- **Preliminary robust screen** — before multivariate GLM,
  discriminant analysis, MANOVA.
- **Outlier detection** — `robustbase::adjbox` / `EllipticEnvelope`
  wrap MCD for practical use.
- **Robust regression preprocessor** — deletion of high-
  leverage points identified by MCD-Mahalanobis distance.

## When NOT to use

- **Very high-dim (p > n / 2)** — MCD is undefined; use
  regularised alternatives (OGK, MRCD, Stahel-Donoho).
- **When clean data is expected** — MLE is more efficient.
- **Non-elliptical distributions** — MCD assumes elliptical
  contours; use OGK / SDE for skewed data.

## Assumptions & caveats

- **α (trimming)** — 0.5 max breakdown, 0.75 higher
  efficiency; robustbase default is 0.5.
- **Consistency correction factor** — MCD covariance is
  scaled by a Gaussian-consistency constant so it estimates
  Σ, not `(h/n) · Σ`.
- **Reweighting step** — Rousseeuw-Van Driessen do a
  reweighted MLE on the h-subset for extra efficiency.
- **Curse of dimensionality** — MCD accuracy degrades for
  p > 20; use OGK (Orthogonalised Gnanadesikan-Kettenring)
  or Deterministic MCD.
- **Randomness** — different seeds give different starts;
  use enough elemental sub-samples.

## Related in this repo

- `least-trimmed-squares` — univariate / regression cousin.
- `mm-estimators-robust`, `mahalanobis-distance-matching`,
  `robust-regression`, `hampel-identifier`,
  `winsorization` — robust neighbours.
- `robust-pca`, `covariance-estimation-highdim` — related
  covariance techniques.
- `multivariate-outlier-detection`, `isolation-forest-anomaly`
  — outlier-detection alternatives.

## Run

```
python techniques/mcd-robust-covariance/python/mcd_robust_covariance.py
Rscript techniques/mcd-robust-covariance/r/mcd_robust_covariance.R
```

**Refs:** Rousseeuw, P.J. "Multivariate estimation with high breakdown point." In *Mathematical Statistics and Applications*, Reidel, 1985; Rousseeuw, P.J. and Van Driessen, K. "A fast algorithm for the minimum covariance determinant estimator." *Technometrics*, 41: 212-223, 1999.

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
