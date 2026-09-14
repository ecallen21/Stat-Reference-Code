# BSTS — Bayesian Structural Time Series (Reference §47.360)

Scott & Varian (2014); Harvey (1989) structural framework.
State-space decomposition of a time series into interpretable
components (level, slope, seasonal, regression), all inferred
jointly under a Bayesian model with spike-and-slab regressor
selection:

```
y_t = μ_t + τ_t + β·x_t + ε_t
μ_t = μ_{t−1} + δ_{t−1} + η_t       (local level with drift)
δ_t = δ_{t−1} + ζ_t                  (slope random walk)
τ_t = −Σ_{k=1..S−1} τ_{t−k} + ξ_t   (dummy seasonal, period S)
```

## Files

- `python/bsts_bayesian_structural_ts.py` — Kalman filter
  with local-linear trend + dummy seasonal (S=12). Synthetic
  series with trend + amplitude-3 sine + amplitude-1.5 half-
  cycle + noise (σ=0.5). Test MAE 0.63, 95% CI coverage
  1.00 for a 12-step forecast.
- `r/bsts_bayesian_structural_ts.R` — `bsts::bsts` (Steven
  Scott's reference implementation), `KFAS`, `dlm` (R);
  `pybsts`, `statsmodels.UnobservedComponents`, from-scratch
  (Python).

## When to use

- **Interpretable forecasts** — separates level, trend,
  seasonality, regression effects.
- **Nowcasting with regressors** — Google Search
  Insights-style nowcasting (the original Scott-Varian
  application).
- **When uncertainty quantification matters** — Bayesian
  filter gives credible intervals directly.
- **Sparse regressors** — spike-and-slab lets BSTS pick
  informative columns automatically.

## When NOT to use

- **Very short series** — Bayesian state-space methods need
  enough data to identify variance components.
- **Non-stationary regime shifts** — additive Gaussian model
  may miss abrupt changes; use regime-switching / HMM.
- **Millisecond forecasts of tick data** — HF settings need
  streaming-optimised models, not full Bayesian filtering.

## Assumptions & caveats

- **Variance priors** — sensitive; the `bsts` package uses
  Inverse-Gamma priors with weakly-informative
  hyperparameters.
- **Model identifiability** — level vs slope trade off; use
  `AddLocalLevel` when a random walk (no slope) is preferred.
- **Regression spike-slab** — `expected.model.size` needs
  tuning; over-shrinks with too small a prior.
- **Kalman filter for point forecasts** — MCMC additionally
  gives posterior over hyperparameters (variance components,
  regression coefficients).

## Related in this repo

- `state-space-kalman`, `state-space-models`,
  `unscented-kalman-filter`, `extended-kalman-filter` —
  the state-space building blocks.
- `arima`, `sarima-arimax`, `holt-winters-forecasting`,
  `theta-method-forecast`, `prophet-forecasting`, `n-beats`,
  `temporal-fusion-transformer` — forecasting alternatives.
- `bayesian-hierarchical-models`, `bayesian-model-averaging`
  — Bayesian modelling neighbours.
- `mrp-poststratification`, `fay-herriot-small-area` — related
  Bayesian small-area frameworks.

## Run

```
python techniques/bsts-bayesian-structural-ts/python/bsts_bayesian_structural_ts.py
Rscript techniques/bsts-bayesian-structural-ts/r/bsts_bayesian_structural_ts.R
```

**Refs:** Scott, S.L. and Varian, H.R. "Predicting the present with Bayesian structural time series." *Int. J. Math. Model. Numer. Optim.*, 5(1-2): 4-23, 2014; Harvey, A.C. *Forecasting, Structural Time Series Models and the Kalman Filter*, Cambridge Univ Press, 1989.

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
