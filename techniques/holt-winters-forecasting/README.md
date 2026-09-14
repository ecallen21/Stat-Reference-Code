# Holt-Winters Triple Exponential Smoothing (Reference §47.309)

Holt (1957); Winters (1960). Decompose a series into LEVEL,
TREND, and SEASONAL components with three smoothing parameters
(α, β, γ):

```
Additive:  l_t = α (y_t − s_{t−m}) + (1 − α)(l_{t−1} + b_{t−1})
           b_t = β (l_t − l_{t−1}) + (1 − β) b_{t−1}
           s_t = γ (y_t − l_{t−1} − b_{t−1}) + (1 − γ) s_{t−m}
           ŷ_{t+h} = l_t + h · b_t + s_{t−m+h_m}
```

Multiplicative version divides by `s`. Classical benchmark
for seasonal series before Prophet / ARIMA.

## Files

- `python/holt_winters_forecasting.py` — Additive HW with
  seasonal period m=12. Simulated series: trend 0.5 / step +
  monthly seasonality + noise. Recovered trend ~0.5; 12-step
  forecast tracks the underlying trend + seasonal pattern.
- `r/holt_winters_forecasting.R` — `forecast::hw`,
  `stats::HoltWinters`, `fable::ETS`, `smooth::es` (R);
  `statsmodels.tsa.holtwinters`, `darts.ExponentialSmoothing`,
  from-scratch (Python).

## When to use

- **Univariate seasonal forecasts** — the workhorse benchmark.
- **Fast, interpretable model** — three parameters,
  transparent decomposition.
- **Batch / online forecasting** — updates are recursive.

## When NOT to use

- **Complex multi-seasonality** — use TBATS / Prophet.
- **Multivariate / cross-series** — VAR / hierarchical
  forecasting.
- **Very noisy short series** — ETS state-space forms handle
  uncertainty better.

## Assumptions & caveats

- **Choose additive vs multiplicative** by variance scaling
  with level.
- **Damped trend** (Gardner-McKenzie 1985) prevents runaway
  extrapolation.
- **α, β, γ tuning** — grid search or MLE; libraries do this
  automatically.
- **Seasonality period m** — must be provided; multi-seasonal
  variants (TBATS) handle multiple.

## Related in this repo

- `arima`, `sarima-arimax` — classical alternatives.
- `prophet-forecasting`, `state-space-kalman` — state-space
  successors.
- `exponential-smoothing` — the level-only precursor.
- `structural-topic-model` (unrelated but neighbouring
  section).

## Run

```
python techniques/holt-winters-forecasting/python/holt_winters_forecasting.py
Rscript techniques/holt-winters-forecasting/r/holt_winters_forecasting.R
```

**Refs:** Holt, C.C. "Forecasting seasonals and trends by exponentially weighted moving averages." *ONR Memorandum 52*, Carnegie Institute of Technology, 1957; Winters, P.R. "Forecasting sales by exponentially weighted moving averages." *Management Science*, 6(3): 324-342, 1960.

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
