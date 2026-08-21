# Functional Time Series (Reference §13.x extra)

A functional TS is a sequence of **curves** `X_t(u)`, `u ∈ [a, b]`, `t = 1, …, T` —
e.g. daily electricity-demand load profile indexed by hour, age-specific
mortality rate by age, yield curve by maturity, or the intraday price
trajectory of an asset.

## Hyndman-Ullah (2007) model

Decompose each curve as mean + a small number of principal-component modes:

```
X_t(u) = μ(u) + Σ_{k=1}^K ξ_{t,k} · φ_k(u) + ε_t(u)
```

- `μ(u)` — mean function.
- `φ_k(u)` — FPCA eigenfunctions from the empirical covariance operator.
- `ξ_{t,k}` — FPCA scores; a K-variate time series.

**Forecast** each score series with a univariate AR (or VAR for correlated
scores), then reconstruct:

```
X̂_{T + h}(u) = μ(u) + Σ_k ξ̂_{T + h, k} · φ_k(u)
```

Discrete implementation: represent each curve on a fixed grid of `D` points;
FPCA reduces to eigendecomposition of the `D × D` sample covariance.

## When to use

- **Mortality forecasting** — the Lee-Carter model is a rank-1 special case; ftsa extends to K ≥ 2 with better fit.
- **Electricity / gas demand** — daily load profiles.
- **Yield-curve forecasting**.
- **Weather / climate** — daily temperature curves, wind profiles.
- **High-frequency intraday finance** — daily price / volume trajectories.
- **Any TS where the observation is naturally a function**, not a scalar.

## Files

- `python/functional_time_series.py` — from-scratch FPCA + AR(2) on scores; rolling one-step forecasts. Demo (T=80 curves, D=24 grid points; mean = 5 + 3 sin(2πu) plus two AR(1) score modes): top-3 FPCs explain 87% + 10% + 0.3%; rolling one-step forecast RMSE 0.385 vs mean-only baseline 0.651 — 40.8% reduction.
- `r/functional_time_series.R` — `ftsa::fts` + `ftsa::ftsm` + `forecast(fitted, h=…)`; `demography::lca` for Lee-Carter; `fda::pca.fd` for classical smoothed FPCA.

## Assumptions & caveats

- **Choice of K** — via variance-explained cutoff (~95%), cross-validation, or scree-plot; too few Ks under-fit dynamics, too many amplify noise into the forecast.
- **AR-on-scores independence assumption** — the scores are cross-sectionally uncorrelated by FPCA construction but the innovations across k can still be correlated; use VAR for joint modelling.
- **Grid choice matters** — a coarse grid loses fine-scale structure; a very fine grid inflates FPCA computational cost with little benefit.
- **Non-stationarity in the mean** — subtract a smooth trend of μ_t(u) before FPCA if there is a level drift over years (mortality data example).
- **Robustness** — outlying curves distort FPCA; use robust FPCA (Locantore / Sinova) or median-based ftsa (`ftsa::rar`).
- **Extrapolation** — FPCA basis is estimated only where curves are observed; forecasts on `u` outside the historical grid are undefined.

## Related in this repo

- `spectral-analysis`, `wavelet-analysis` — frequency-domain summaries of scalar TS.
- `arima`, `var-cointegration` — scalar / vector TS models used on the scores.
- `pca`, `kernel-pca` — scalar-data analogues of FPCA.
- `gaussian-process-regression` — Bayesian alternative for curve modelling.

## Run

```
python techniques/functional-time-series/python/functional_time_series.py
Rscript techniques/functional-time-series/r/functional_time_series.R
```

**Refs:** Hyndman, R.J. & Ullah, M.S. "Robust forecasting of mortality and fertility rates: A functional data approach." *Comput. Stat. Data Anal.* 51(10), 4942–4956, 2007; Ramsay, J.O. & Silverman, B.W. *Functional Data Analysis*, 2nd ed., Springer, 2005; Bosq, D. *Linear Processes in Function Spaces: Theory and Applications*, Springer, 2000.

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
