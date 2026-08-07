# Decomposable Forecasting (Reference §13.21)

Prophet-style additive model (Taylor & Letham 2018):

```
y_t = g(t) + s(t) + h(t) + ε_t
```

where

- **`g(t)`** — trend, usually **piecewise linear** with automatic changepoints,
- **`s(t)`** — seasonality via a **Fourier series** at a fixed period,
- **`h(t)`** — **holiday / event indicators**,
- **`ε_t`** — residual noise.

The strength is that non-statisticians can plug in domain knowledge as regressor columns; the model absorbs it while remaining interpretable.

## Piecewise-linear trend

Pick `S` candidate changepoints uniformly over the training period. Fit

```
g(t) = k · t + m + Σ_s δ_s (t − s)_+
```

Regularize `δ` with an L1 (Laplace) penalty so most `δ_s ≈ 0` and only a few real changepoints survive.

## Fourier seasonality

```
s(t) = Σ_{k=1}^K [ a_k cos(2π k t / P) + b_k sin(2π k t / P) ]
```

`K` controls smoothness; typical defaults: `K = 3` (weekly), `K = 10` (yearly).

## Holidays

Binary regressors, one per named holiday (optionally with a small lead/lag window).

## Files

- `python/decomposable_forecasting.py` — trend + Fourier + holiday indicators fit jointly by ridge regression. Not a full Prophet re-implementation, but produces essentially the same shape on smooth series. Demo (n = 365, noise SD 1.0): in-sample RMSE = 0.99, recovers trend, weekly seasonality, and three holiday spikes.
- `r/decomposable_forecasting.R` — wraps `prophet::prophet` (installed via `install.packages("prophet")`).

## When to use

- Retail / marketing time series with recurring weekly and yearly cycles.
- Any series where holidays / promotions produce localized jumps that should be attributed rather than absorbed into noise.
- When you need **interpretable component plots** — trend / seasonality / holidays split out separately.

## When NOT to use

- Short series (< 2 seasonal cycles) — the Fourier basis overfits.
- Series with strong autocorrelation in `ε_t` — combine with an ARIMA on residuals or switch to ETS / SARIMA.
- Extremely irregular sampling — the basis assumes evenly-spaced time.

## Assumptions & caveats

- **Additive vs multiplicative**: swap `y_t = g(t) · s(t) · h(t) · ε_t` (i.e. log-transform `y` first) when seasonal amplitude grows with the trend.
- **Changepoint prior scale**: Prophet's `changepoint_prior_scale` controls how flexible the trend is; too large overfits, too small misses real breaks.
- **Uncertainty intervals**: Prophet's are simulation-based and can be over-optimistic; consider a proper Bayesian TS model for calibrated intervals.

## Run

```
python techniques/decomposable-forecasting/python/decomposable_forecasting.py
Rscript techniques/decomposable-forecasting/r/decomposable_forecasting.R
```

**Refs:** Taylor, S.J. & Letham, B. "Forecasting at scale." *Am. Stat.* 72(1), 37–45, 2018 (Prophet); Harvey, A.C. *Forecasting, Structural Time Series Models and the Kalman Filter*, Cambridge, 1989.

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
