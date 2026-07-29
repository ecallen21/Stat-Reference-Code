# ARIMA Models (Reference §13.4, §13.5, §13.52)

The classical time-series forecasting workhorse. `ARIMA(p, d, q)` = ARMA on the `d`-th difference:

```
AR(p) : x_t = c + φ_1 x_{t-1} + ... + φ_p x_{t-p} + ε_t
MA(q) : x_t = c + ε_t + θ_1 ε_{t-1} + ... + θ_q ε_{t-q}
ARIMA(p, d, q) : combine + apply to Δ^d x
```

## Model identification workflow

1. **Stationarity**: check with ADF / KPSS (`stationarity-tests`); difference `d` times until stationary.
2. **PACF cuts off at lag p** → AR(p) suggested.
3. **ACF cuts off at lag q** → MA(q) suggested.
4. **Both taper** → ARMA(p, q); grid-search by AIC / BIC.

## Order selection (§13.52)

Fit models over a grid of `(p, d, q)` and keep the smallest **AIC** or **BIC**. AIC picks slightly larger models (better forecast); BIC is more parsimonious.

## Residual diagnostic

Ljung-Box on residuals — large p means residuals look like white noise → model captured the autocorrelation structure adequately.

## Files

- `python/arima.py` — from-scratch conditional-SS MLE for ARMA via BFGS + `fit_arima(p, d, q)` + `auto_order()` AIC grid + `ljung_box_residuals()`. ARMA(1,1) coefficients match `statsmodels.tsa.arima.model.ARIMA` to 3 dp on the demo.
- `r/arima.R` — thin wrappers around base `stats::arima` and `forecast::auto.arima`.

## Assumptions

- Weak stationarity (after differencing).
- Independent Gaussian innovations.
- Correct `(p, d, q)` — inspect residual ACF / PACF + Ljung-Box.

## Run

```
python techniques/arima/python/arima.py
Rscript techniques/arima/r/arima.R
```

**Refs:** Box, G.E.P. & Jenkins, G.M. *Time Series Analysis: Forecasting and Control*, Holden-Day, 1970; Hyndman, R.J. & Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed., OTexts, 2021 (Ch. 8–9).

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
