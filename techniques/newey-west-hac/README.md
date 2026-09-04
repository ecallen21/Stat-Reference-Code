# Newey-West HAC + Cluster-Robust SEs (Reference §35.15)

OLS standard errors are wrong under **heteroscedasticity** (White
1980), **autocorrelation** (HAC / Newey-West 1987), or **clustered
errors** (Liang-Zeger 1986, Cameron-Miller 2015).

## Newey-West HAC

```
V_HAC  =  (X'X)⁻¹ Ω̂ (X'X)⁻¹
Ω̂ = Σ_i x_i x_iᵀ u_i²
     + Σ_{l=1..L} w_l · Σ_i (x_i x_{i+l}ᵀ + x_{i+l} x_iᵀ) u_i u_{i+l}
Bartlett weights: w_l = 1 − l / (L + 1)
```

Bandwidth `L ≈ 4 (n / 100)^(2/9)` (Newey-West rule of thumb).

## Cluster-robust

```
V_CR  =  (X'X)⁻¹ (Σ_g X_gᵀ u_g u_gᵀ X_g) (X'X)⁻¹
```

Small-sample correction `(G / (G−1)) · ((n−1)/(n−k))` (Cameron-
Gelbach-Miller).

## When to use

- **Time-series regression** with autocorrelated residuals → HAC.
- **Panel / clustered sampling** (schools, hospitals, firms) →
  cluster-robust.
- **Cross-section with heteroscedasticity only** → White HC1 / HC3.

## When NOT to use

- **Small number of clusters (< 20)** — cluster-robust is
  anti-conservative; use wild-cluster bootstrap or CR2 (Bell-McCaffrey).
- **Extreme serial correlation** — HAC with Bartlett kernel has slow
  convergence; consider Kiefer-Vogelsang tests.

## Files

- `python/newey_west_hac.py` — from-scratch OLS + Newey-West HAC +
  cluster-robust SEs. Demo:
  - **AR(1) errors (n=300)**: HAC intercept SE = 0.12 vs OLS = 0.07
    (70 % larger); slope similar (`x` is iid).
  - **Clustered errors (30 × 10)**: cluster-robust slope SE differs
    from OLS.
- `r/newey_west_hac.R` — `sandwich::vcovHAC / vcovCL`,
  `clubSandwich`, `lmtest::coeftest` (R); `statsmodels`,
  `linearmodels` (Python).

## Assumptions & caveats

- **Bandwidth `L`** — auto-select (Andrews 1991) or rule-of-thumb.
- **Kernel choice** — Bartlett (used here), Parzen, Quadratic-Spectral.
- **Two-way clustering** (e.g., firm × year) — Cameron-Miller 2011.
- **HAR / IM-OLS** — small-bandwidth-robust tests (Kiefer-Vogelsang
  2005).
- **Driscoll-Kraay** for panel data with cross-sectional dependence.

## Related in this repo

- `sandwich-robust-se` — general Huber-White SEs (already in repo).
- `event-study`, `staggered-did`, `fixed-effects-panel` — panel
  methods that need cluster-robust SEs.
- `iv-2sls`, `gmm-general` — need HAC / cluster-robust SEs for valid
  inference.

## Run

```
python techniques/newey-west-hac/python/newey_west_hac.py
Rscript techniques/newey-west-hac/r/newey_west_hac.R
```

**Refs:** Newey, W.K. & West, K.D. "A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix." *Econometrica*, 1987; Cameron, A.C. & Miller, D.L. "A practitioner's guide to cluster-robust inference." *Journal of Human Resources*, 2015.

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
