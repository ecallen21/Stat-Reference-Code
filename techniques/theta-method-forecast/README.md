# Theta Method (Reference §47.349)

Assimakopoulos & Nikolopoulos (2000). Decompose a series `y_t`
into two THETA LINES parameterised by `θ`:

```
θ-line(θ, t) = y_t + (θ − 1)/2 · sum-of-second-differences
```

Standard variant averages `θ = 0` (linear trend) and `θ = 2`
(short-term features with doubled curvature). Forecast each
component separately (linear extrapolation + SES) and average.
Winner of the M3 competition among simple statistical models
(Makridakis-Hibon 2000).

## Files

- `python/theta_method_forecast.py` — synthetic level +
  trend + seasonal series (n=60, h=12). Theta test-MAE 4.18
  vs naive last-value 5.65 and pure linear extrapolation
  3.16 (linear is close because signal is dominated by a
  linear trend; theta averages both effects).
- `r/theta_method_forecast.R` — `forecast::thetaf`,
  `forecTheta::stheta` (R); `sktime.ThetaForecaster`,
  `statsforecast.Theta`, from-scratch (Python).

## When to use

- **Univariate monthly / quarterly forecasts** — competitive
  baseline in M3 / M4 competitions.
- **Sparse-signal series** where ARIMA over-fits.
- **When simplicity + reliability matter** more than a
  marginal accuracy edge.

## When NOT to use

- **Strong seasonality without decomposition** — do STL/
  seasonal-adjust first, then theta the residual.
- **Cross-sectional or panel data** — theta is univariate.
- **Long horizons** — trend uncertainty compounds; use
  probabilistic models.

## Assumptions & caveats

- **SES smoothing α** — tuned via MSE minimisation (0.3 is a
  common default).
- **Trend line = OLS on time** — sensitive to outliers; use
  Theil-Sen slope for robustness.
- **Multiplicative variant** — Theta-B, Theta-BC transform
  first (Bergmeir-Hyndman-Koo 2016).
- **Bootstrap prediction intervals** — needed for uncertainty,
  since the point-forecast method is deterministic.

## Related in this repo

- `exponential-smoothing`, `holt-winters-forecasting` —
  SES / trend / seasonal cousins.
- `arima`, `sarima-arimax`, `prophet-forecasting` — classical
  forecasting alternatives.
- `deepar-probabilistic-forecast`, `n-beats`,
  `temporal-fusion-transformer` — neural forecasting cousins.
- `forecast-combination`, `forecast-evaluation-cv` —
  aggregation and validation.

## Run

```
python techniques/theta-method-forecast/python/theta_method_forecast.py
Rscript techniques/theta-method-forecast/r/theta_method_forecast.R
```

**Refs:** Assimakopoulos, V. and Nikolopoulos, K. "The theta model: a decomposition approach to forecasting." *Int. J. Forecast.*, 16(4): 521-530, 2000.

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
