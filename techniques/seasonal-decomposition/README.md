# Seasonal-Trend Decomposition (Reference §13.24, §13.47, §13.54)

Split a time series into three parts:

```
Additive:        y_t  =  T_t  +  S_t  +  R_t
Multiplicative:  y_t  =  T_t  ·  S_t  ·  R_t

T = trend-cycle,  S = seasonal,  R = remainder
```

Additive when the seasonal amplitude is roughly constant; multiplicative when it scales with the level.

## Three approaches

| Method | Trend | Seasonal | Handles changing seasonality? | Robust to outliers? |
|---|---|---|---|---|
| **Classical** (§13.47) | Centered moving average | Averaged detrended values | No | No |
| **STL** (§13.24; Cleveland 1990) | LOESS smooth | LOESS on each seasonal position | **Yes** | With `robust=True` |
| **X-13ARIMA-SEATS** (§13.54; US Census) | Reg-ARIMA + SEATS | Signal extraction | Yes | Yes |

## When to use each

- **Classical** — quick look, simple data, single seasonality, no outliers.
- **STL** — the modern default. Fast, flexible, handles evolving seasonality.
- **X-13** — official statistics (BLS, ONS). Requires the external X-13 binary; use `seasonal::seas()` in R.

## Files

- `python/seasonal_decomposition.py` — from-scratch classical additive + multiplicative + STL via `statsmodels.tsa.seasonal.STL`. Recovers trend 13.11 vs. true 13.0 and remainder SD 0.40 (exact) on the demo.
- `r/seasonal_decomposition.R` — base `stats::decompose` + `stats::stl` + `seasonal::seas` for X-13.

## Assumptions

- Regularly-spaced observations.
- Seasonal period `m` known / correctly specified.
- Additive vs. multiplicative choice matches how the seasonal amplitude scales.

## Run

```
python techniques/seasonal-decomposition/python/seasonal_decomposition.py
Rscript techniques/seasonal-decomposition/r/seasonal_decomposition.R
```

**Refs:** Cleveland, R.B., Cleveland, W.S., McRae, J.E. & Terpenning, I. "STL: a seasonal-trend decomposition procedure based on loess." *J. Off. Stat.* 6(1), 3–73, 1990; Findley, D.F. "Some recent developments and directions in seasonal adjustment." *J. Off. Stat.* 21(2), 343–365, 2005 (X-13ARIMA-SEATS); Hyndman, R.J. & Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed., OTexts, 2021 (Ch. 3).

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
