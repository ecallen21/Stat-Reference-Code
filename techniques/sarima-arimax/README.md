# SARIMA + ARIMAX / Transfer Function (Reference §13.6, §13.25)

Two extensions of ARIMA covering the most common real-world needs.

## SARIMA(p, d, q)(P, D, Q)_s (§13.6)

Seasonal ARIMA. Adds a second set of orders that operate on **lag-s** dynamics for period `s`:

- `(p, d, q)` — non-seasonal orders (lag-1 dynamics).
- `(P, D, Q)` — seasonal orders (lag-s dynamics).
- `s` — seasonal period (12 for monthly-annual, 4 for quarterly-annual, 7 for daily-weekly, 24 for hourly-daily, …).

Example: `SARIMA(1, 1, 1)(1, 1, 1)_{12}` = first-differences plus seasonal-differences, with AR(1) + MA(1) at both lags.

## ARIMAX / Regression with ARIMA Errors (§13.25)

```
y_t  =  Z_t' β  +  u_t         where u_t ~ ARIMA(p, d, q)
```

Interpretation: `Z_t β` captures known drivers (temperature, holidays, promotions, Fourier-seasonal terms). The ARIMA part absorbs the leftover autocorrelated noise. Same as **regression with ARIMA errors**.

## Transfer function models

Generalize ARIMAX by allowing the exogenous input to propagate through its own ARMA-style dynamics — e.g., a marketing spend today affects sales over the next several weeks with a specific decay. `statsmodels.SARIMAX` supports this via the `exog` interface; specialized software (SAS PROC TRANSFER, R's `TSA::arimax`) handles more elaborate specifications.

## Files

- `python/sarima_arimax.py` — thin wrappers around `statsmodels.tsa.statespace.sarimax.SARIMAX` for both SARIMA and ARIMAX; returns fitted params, AIC/BIC, and 12-step forecasts.
- `r/sarima_arimax.R` — thin wrappers around base `stats::arima` (with `seasonal =` and `xreg =`) + `forecast::auto.arima` for automatic order selection.

## Assumptions

- Same as ARIMA (stationarity after differencing; Gaussian innovations).
- Seasonal period `s` known / chosen correctly (usually obvious from the data — but multiple seasonalities need TBATS / Fourier-ARIMAX).
- For ARIMAX: `Z_t` must be available in the forecast horizon or itself forecastable.

## Run

```
python techniques/sarima-arimax/python/sarima_arimax.py
Rscript techniques/sarima-arimax/r/sarima_arimax.R
```

**Refs:** Box, G.E.P., Jenkins, G.M. & Reinsel, G.C. *Time Series Analysis: Forecasting and Control*, 4th ed., Wiley, 2008 (Ch. 9, 12); Hyndman, R.J. & Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed., OTexts, 2021 (Ch. 8.9, 10).

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
