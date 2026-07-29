# Stationarity Tests: ADF, KPSS, Phillips-Perron (Reference §13.2; also covers §13.8, §13.53)

Most classical time-series methods assume **stationarity**: constant mean and variance, autocovariance depending only on lag.

## Three tests with different nulls

| Test | H₀ | Small p means | Approach |
|---|---|---|---|
| **ADF** (Augmented Dickey-Fuller) | Unit root (non-stationary) | Stationary | Regression on `Δx_t = α + βt + ρx_{t-1} + lagged Δx's` |
| **KPSS** (Kwiatkowski et al.) | Stationary | Non-stationary | Residual variance / long-run variance |
| **Phillips-Perron** | Unit root | Stationary | Nonparametric autocorrelation correction to ADF |

## §13.53 Reconciling ADF and KPSS

Because the tests have **opposite nulls**, both can be run for a stronger conclusion:

| ADF rejects | KPSS rejects | Verdict |
|---|---|---|
| Yes | No | **STATIONARY** (both agree) |
| No | Yes | **NON-STATIONARY** (both agree) |
| Yes | Yes | Inconsistent — likely near-stationary; try differencing |
| No | No | Inconclusive — low power; more data / different test |

Rule of thumb: if non-stationary, **difference once** and re-test. Repeat until both agree on stationarity — that's your `d` for `ARIMA(p, d, q)`.

## Files

- `python/stationarity_tests.py` — thin wrappers around `statsmodels.tsa.stattools.{adfuller, kpss}` + `arch.unitroot.PhillipsPerron` (optional) + `reconcile_adf_kpss()` combined-verdict helper + `difference()`.
- `r/stationarity_tests.R` — thin wrappers around `tseries::{adf.test, kpss.test, pp.test}`.

## Assumptions

- Regularly-sampled time series.
- Autocorrelation structure matters — pick a max lag that captures relevant memory (default: automatic via AIC in `adfuller`, `nlags="auto"` in KPSS).
- Sample size ≥ 50 for reliable tests; larger for weak trends.

## Run

```
python techniques/stationarity-tests/python/stationarity_tests.py
Rscript techniques/stationarity-tests/r/stationarity_tests.R
```

**Refs:** Dickey, D.A. & Fuller, W.A. "Distribution of the estimators for autoregressive time series with a unit root." *JASA* 74(366a), 427–431, 1979; Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. & Shin, Y. "Testing the null hypothesis of stationarity against the alternative of a unit root." *J. Econom.* 54(1–3), 159–178, 1992; Phillips, P.C.B. & Perron, P. "Testing for a unit root in time series regression." *Biometrika* 75(2), 335–346, 1988.

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
