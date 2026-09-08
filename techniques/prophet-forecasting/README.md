# Prophet Forecasting (Reference §47.116)

Taylor & Letham (2018). Additive decomposable model:

    y(t) = g(t) + s(t) + h(t) + ε

    * g(t) piecewise-linear trend with automatic change-points
    * s(t) Fourier-series seasonality
    * h(t) user-supplied holiday effects
    * ε    IID noise.

Bayesian model in Stan by default; analyst-friendly forecasting for
business KPIs with missing / irregular / outlier-prone daily series.

## Files

- `python/prophet_forecasting.py` — least-squares fit of a
  changepoint-piecewise trend + Fourier seasonality basis
  (simplified Prophet). Demo (n=240, piecewise trend + period-30
  seasonality + noise, train 200 / test 40):
  - training RMSE 0.28
  - **Prophet forecast RMSE 0.38**
  - Persistence baseline RMSE 2.75
  - Seasonal-naive baseline RMSE 1.05.
- `r/prophet_forecasting.R` — `prophet` (Facebook, R + Python);
  `fable::PROPHET`, `neuralprophet` (Python).

## When to use

- **Business-KPI daily / weekly forecasts** — sales, traffic,
  bookings.
- **Analyst-friendly** with domain-relevant holiday effects.
- **Missing values / outliers** handled automatically.
- **Change-point detection built-in**.

## When NOT to use

- **Short high-frequency series** (sub-daily, high-freq finance) —
  use ARIMA / state-space / RNN.
- **When you need probabilistic ensembles** — modern deep-forecast
  ecosystem (N-BEATS, TFT) may win.
- **Hierarchical / multi-series** — `fable`, `hierarchical-forecasting`
  offer coherent reconciliation.

## Assumptions & caveats

- **Additive decomposition** — for multiplicative seasonality use
  a log transform first.
- **Change-point priors** — `changepoint_prior_scale` too small →
  underfits trend shifts; too large → overfit / wobbly.
- **Fourier order K** — higher order captures more seasonality
  detail; risk of overfitting short cycles.
- **No autoregression by default** — add ARIMA errors (Prophet R has
  `arima_order`) when residuals show autocorrelation.

## Related in this repo

- `arima`, `sarima-arimax`, `arfima`, `exponential-smoothing`,
  `state-space-models`, `state-space-kalman`,
  `seasonal-decomposition`, `hierarchical-forecasting`,
  `forecast-combination`, `forecast-evaluation-cv` — TS
  cousins.
- `structural-breaks-its`, `change-point-detection`,
  `bai-perron-multiple-breaks` — change-point neighbours.
- `bayesian-hierarchical-models`, `hmc-nuts`, `laplace-approximation`
  — Bayesian back-end used by full Prophet.

## Run

```
python techniques/prophet-forecasting/python/prophet_forecasting.py
Rscript techniques/prophet-forecasting/r/prophet_forecasting.R
```

**Refs:** Taylor, S.J. & Letham, B. "Forecasting at scale." *The American Statistician* 72(1): 37-45, 2018.

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
