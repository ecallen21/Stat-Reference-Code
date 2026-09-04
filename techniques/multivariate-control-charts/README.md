# Multivariate Control Charts (Reference §37.4)

Hotelling (1947). For p-dim observations `x_i`, monitor

```
T_i²  =  (x_i − μ_0)ᵀ Σ_0⁻¹ (x_i − μ_0)   ~   F  (up to a scaling)
```

Detects out-of-control conditions that univariate charts on
individual variables miss.

## Phase-II UCL (individual observations)

```
UCL  =  p (n + 1)(n − 1) / (n (n − p))  ·  F_{α; p, n − p}.
```

Signal when `T² > UCL`.

## Variants

- **MEWMA** — multivariate EWMA (Lowry et al. 1992).
- **MCUSUM** — multivariate CUSUM (Crosier 1988).
- **Regression-adjusted T²** — subtract predictions from process
  variables to isolate residual variation.

## When to use

- **Correlated quality variables** — chemistry, sensor arrays, flight
  telemetry.
- **Missed univariate shifts** where movement is in a joint direction
  (rotation in feature space).

## When NOT to use

- **Independent variables** — univariate Shewhart works fine.
- **Very high p** — inverse of Σ becomes ill-conditioned; use PCA
  monitoring first.

## Files

- `python/multivariate_control_charts.py` — from-scratch Hotelling
  T² + F-based UCL. Demo p=3 correlated variables, Phase I n=100
  baseline; Phase II 20 in-control + 10 obs shifted only in x₂.
  **UCL(α=0.005) = 14.07**; 3 of 10 shifted obs flagged (T² 21, 20,
  18). No in-control obs breaches UCL.
- `r/multivariate_control_charts.R` — `qcc::mqcc`, `MSQC` (R);
  `multivariate-quality-control` (Python).

## Assumptions & caveats

- **Multivariate normality** — check with Mahalanobis Q-Q plot.
- **Σ estimation** — needs `n > p`; use shrinkage (Ledoit-Wolf) for
  `p ≈ n`.
- **Decomposition of a signal** — MYT decomposition attributes T²
  contribution to individual variables (Mason-Young-Tracy 1995).
- **Phase I control** — establish Σ_0 from in-control data; contaminated
  Phase I data biases limits.

## Related in this repo

- `shewhart-control-charts`, `cusum-charts`, `ewma-charts` — univariate
  siblings.
- `pca`, `sparse-pca` — dimension reduction before T² for large p.
- `robust-pca`, `covariance-estimation-highdim` — for
  outlier-contaminated Phase I.
- `process-capability-indices`, `six-sigma-methods` — sibling SPC
  tools.

## Run

```
python techniques/multivariate-control-charts/python/multivariate_control_charts.py
Rscript techniques/multivariate-control-charts/r/multivariate_control_charts.R
```

**Refs:** Hotelling, H. "Multivariate quality control." *Techniques of Statistical Analysis*, 1947; Mason, R.L. & Young, J.C. *Multivariate Statistical Process Control with Industrial Applications*, SIAM, 2002.

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
