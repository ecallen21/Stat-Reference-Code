# Exponential Smoothing Family (Reference §13.3; also covers §13.43, §13.56)

Weight past observations with **exponentially decaying importance**. Three main variants:

## SES — Simple Exponential Smoothing

Flat forecast; suits stationary series with no trend or seasonality.
```
l_t = α · y_t + (1 − α) · l_{t-1}
forecast: ŷ_{t+h} = l_t         (flat)
```

## Holt — Level + Trend

Linear-trend forecast; suits trending non-seasonal series.
```
l_t = α · y_t + (1 − α)(l_{t-1} + b_{t-1})
b_t = β · (l_t − l_{t-1}) + (1 − β) · b_{t-1}
forecast: ŷ_{t+h} = l_t + h · b_t
```

## Holt-Winters — Level + Trend + Seasonal

Additive or multiplicative seasonal; suits seasonal series with trend.

**Additive** (this file):
```
l_t = α · (y_t − s_{t-m}) + (1 − α)(l_{t-1} + b_{t-1})
b_t = β · (l_t − l_{t-1}) + (1 − β) · b_{t-1}
s_t = γ · (y_t − l_{t-1} − b_{t-1}) + (1 − γ) · s_{t-m}
```

## §13.43 ETS and TBATS

**ETS** = Error / Trend / Seasonal — unified state-space specification of the exponential-smoothing family with likelihood-based parameter estimation. `statsmodels.tsa.holtwinters.ExponentialSmoothing` and R's `forecast::ets` fit it.

**TBATS** = Trigonometric Box-Cox ARMA Trend Seasonal — supports **multiple seasonalities** (e.g. daily + weekly + yearly for hourly retail data). R's `forecast::tbats`.

## §13.56 Which to pick

- **SES** — no trend, no seasonality.
- **Holt** — trend, no seasonality.
- **Holt-Winters** — trend + a single seasonality.
- **ETS** — let the software choose the ETS combination via AIC.
- **TBATS** — multiple / complex seasonalities.

## Files

- `python/exponential_smoothing.py` — from-scratch SES / Holt / Holt-Winters additive with SSE-based parameter selection; cross-check `statsmodels.tsa.holtwinters.ExponentialSmoothing`.
- `r/exponential_smoothing.R` — base `stats::HoltWinters` + `forecast::ets` + `forecast::tbats`.

## Assumptions

- Regularly-spaced observations.
- Trend / seasonality choice matches the data — inspect first.
- For multi-step forecasts, prediction intervals require a state-space (ETS) fit; SSE-based fits don't produce them directly.

## Run

```
python techniques/exponential-smoothing/python/exponential_smoothing.py
Rscript techniques/exponential-smoothing/r/exponential_smoothing.R
```

**Refs:** Holt, C.C. "Forecasting seasonals and trends by exponentially weighted moving averages." *ONR Res. Memo* 52, 1957 (reprinted *IJF* 20, 2004); Winters, P.R. "Forecasting sales by exponentially weighted moving averages." *Mgmt. Sci.* 6(3), 324–342, 1960; Hyndman, R.J., Koehler, A.B., Snyder, R.D. & Grose, S. "A state space framework for automatic forecasting using exponential smoothing methods." *IJF* 18(3), 439–454, 2002; De Livera, A.M., Hyndman, R.J. & Snyder, R.D. "Forecasting time series with complex seasonal patterns using exponential smoothing." *JASA* 106(496), 1513–1527, 2011.

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
