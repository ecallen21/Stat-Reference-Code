# ACF, PACF, Ljung-Box, CCF, Mann-Kendall (Reference §13.1; also covers §13.9, §13.42, §13.48)

The five essential time-series diagnostic tools bundled in one file.

## What each one does

| Function | What it measures | Use for |
|---|---|---|
| **ACF** (§13.1) | Correlation of `x_t` with `x_{t-k}` | Model identification; detecting seasonality / MA structure |
| **PACF** (§13.1) | Correlation after partialling out intermediate lags | AR-order identification (cuts off at lag p for AR(p)) |
| **Ljung-Box** (§13.42) | Joint test of no autocorrelation over lags 1..h | Residual diagnostic after fitting a model |
| **CCF** (§13.48) | Cross-correlation between two series at lags −h..+h | Lead-lag structure between series X and Y |
| **Mann-Kendall** (§13.9) | Non-parametric monotone trend | Trend test when data isn't Gaussian / linear |

## Formulas

```
ACF: ρ̂(k) = Σ_{t=k+1}^n (x_t − x̄)(x_{t−k} − x̄) / Σ_t (x_t − x̄)²

PACF: solve the Yule-Walker equations at each lag k;
      last coefficient of the AR(k) fit is φ̂_kk = PACF(k)

Ljung-Box: Q = n(n+2) Σ_{k=1}^h ρ̂(k)² / (n − k)   ~ χ²_h under H₀
```

## Files

- `python/acf_pacf.py` — from-scratch `acf`, `pacf` (Yule-Walker), `ljung_box`, `ccf`, `mann_kendall`. Cross-check `statsmodels.tsa.stattools.{acf, pacf}`.
- `r/acf_pacf.R` — thin wrappers around `stats::{acf, pacf, ccf, Box.test}` + `Kendall::MannKendall`.

## Assumptions

- Regularly-spaced observations (no missing timestamps). For irregular data, use time-domain interpolation or specialized irregular-TS methods.
- Sample ACF/PACF have `~ 1/√n` SE — the ± 1.96/√n confidence band assumes weak stationarity.
- Mann-Kendall tests monotonicity, not linearity — a step or exponential trend registers as a trend.

## Run

```
python techniques/acf-pacf/python/acf_pacf.py
Rscript techniques/acf-pacf/r/acf_pacf.R
```

**Refs:** Box, G.E.P., Jenkins, G.M. & Reinsel, G.C. *Time Series Analysis: Forecasting and Control*, 4th ed., Wiley, 2008; Ljung, G.M. & Box, G.E.P. "On a measure of a lack of fit in time series models." *Biometrika* 65(2), 297–303, 1978; Kendall, M.G. *Rank Correlation Methods*, 4th ed., Griffin, 1975.

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
