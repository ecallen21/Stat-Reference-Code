# Heteroscedasticity- and Cluster-Robust Standard Errors (Reference §5.7, §5.8)

The OLS coefficient variance is

```
Cov(β̂) = (XᵀX)⁻¹ Xᵀ Ω X (XᵀX)⁻¹                (sandwich form)
```

Under homoscedasticity `Ω = σ² I` and this collapses to the usual `(XᵀX)⁻¹ σ²`. Under heteroscedasticity or clustering, `Ω` is not scalar; classical SEs are **wrong** — sometimes too small, sometimes too large.

## Heteroscedasticity-consistent (HC) sandwich (White 1980)

Plug in `Ω̂` from the observed residuals:

| Estimator | `Ω̂_{ii}`                                | Notes                                       |
|-----------|---------------------------------------|---------------------------------------------|
| HC0       | `r_i²`                                | White's original; biased small-sample       |
| HC1       | `n / (n − k) · r_i²`                  | Stata default                               |
| HC2       | `r_i² / (1 − h_ii)`                   | Leverage-adjusted                           |
| **HC3**   | `(r_i / (1 − h_ii))²`                 | MacKinnon-White 1985; recommended default   |

`h_ii = X_i (XᵀX)⁻¹ X_iᵀ` is the leverage of observation `i`. **Report HC3 by default** for `n ≤ ~250`; HC1 and HC3 agree in large samples.

## Cluster-robust (Liang-Zeger 1986)

Errors correlated within group `g`:

```
Ω̂ = Σ_g X_gᵀ r_g r_gᵀ X_g       (sum of cluster-level score outer products)
```

Standard finite-sample scaling: multiply by `(G / (G − 1)) · ((n − 1) / (n − k))` (Stata convention).

## Files

- `python/sandwich_robust_se.py` — HC0 / HC1 / HC3 + cluster-robust from scratch. Demo: cluster-robust SE of intercept = 0.230 (classical 0.076, HC3 0.077) on 40 clusters of 10 with cluster-level shocks — captures the ~3× SE inflation. Matches `statsmodels.OLS(...).fit(cov_type = "cluster"/"HC3")` to 4 decimals.
- `r/sandwich_robust_se.R` — `sandwich::vcovHC(type = "HC3")` and `sandwich::vcovCL(cluster = ...)` + `lmtest::coeftest`.

## When to use

- **HC3** — any regression where heteroscedasticity is plausible: continuous outcomes with variance related to `x`, cross-sectional studies with heterogeneous groups, count regressions.
- **Cluster-robust** — repeated measures on the same subject, students within schools, patients within hospitals, firms within industries.
- **Two-way cluster** — panel data with correlation across time within firm and across firms within time (Cameron-Gelbach-Miller 2011).

## When NOT to use

- **Purely homoscedastic** data — classical SE is efficient; robust SE just adds noise.
- **Few clusters** (`G < 30`) — cluster-robust SEs are **downward-biased**; use wild-cluster bootstrap (Cameron-Gelbach-Miller 2008) or T(G-1) reference distribution.
- **Nested-multi-level structure** — a mixed-effects model gives efficient and correctly-shaped inference; robust SE only fixes the standard error.

## Diagnostics

- **Breusch-Pagan / White test** — formal test for heteroscedasticity.
- **Residual vs fitted plot** — visual.
- **HC0 vs HC3 comparison** — if they agree, small-sample bias isn't a problem.

## Run

```
python techniques/sandwich-robust-se/python/sandwich_robust_se.py
Rscript techniques/sandwich-robust-se/r/sandwich_robust_se.R
```

**Refs:** White, H. "A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity." *Econometrica* 48(4), 817–838, 1980; Liang, K.-Y. & Zeger, S.L. "Longitudinal data analysis using generalized linear models." *Biometrika* 73(1), 13–22, 1986; MacKinnon, J.G. & White, H. "Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties." *J. Econometrics* 29(3), 305–325, 1985.

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
