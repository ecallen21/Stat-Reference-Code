# Forecast Evaluation: TS CV + Accuracy Metrics + Reconciliation (Reference §13.23; also covers §13.31, §13.35, §13.36, §13.45, §13.51)

## §13.23 Time-Series Cross-Validation

Standard k-fold CV **leaks future into past** — completely invalid for time series. Two valid alternatives:

- **Expanding window**: fit on `[1..t]`, test on `[t+1..t+h]`, grow `t` by `step`.
- **Rolling origin (fixed window)**: fit on `[t−w+1..t]`, test on `[t+1..t+h]`, slide `t`.

Both give a distribution of forecast errors that respects temporal ordering.

## §13.31 / §13.51 Accuracy metrics

| Metric | Formula | Notes |
|---|---|---|
| **MAE** | `mean(|y − ŷ|)` | Units of y; interpretable |
| **RMSE** | `√mean((y − ŷ)²)` | Units of y; penalizes big misses |
| **MAPE** | `mean(|y − ŷ| / |y|) · 100%` | %; blows up when y ≈ 0 |
| **sMAPE** | `mean(2·|y − ŷ| / (|y| + |ŷ|)) · 100%` | Symmetric; bounded |
| **MASE** | `MAE / in-sample-naive-MAE` | Scale-free; 1 = as good as naive; **the M4 standard** |
| **CRPS** | Probabilistic (density- or ensemble-based) | For predictive distributions |

## §13.35 Multi-step forecasting strategies

- **Recursive**: fit one-step model; feed forecasts back for next step. Error accumulates.
- **Direct**: separate model for each horizon `h`. Independent; no error propagation.
- **Hybrid**: recursive for short horizons, direct for long. Compromise.

## §13.45 Hierarchical reconciliation

When forecasts must sum (regions → country; products → category):

| Method | Description |
|---|---|
| **Bottom-up** | Sum base-level forecasts. Simple; loses info from top-level model. |
| **Top-down** | Allocate top forecast by historical proportions. |
| **MinT** (Wickramasuriya et al. 2019) | Minimum-trace optimal reconciliation; combines all levels. |

## Files

- `python/forecast_evaluation_cv.py` — all metrics (MAE / RMSE / MAPE / sMAPE / MASE) + `expanding_window_cv()` + bottom-up hierarchical reconciliation.
- `r/forecast_evaluation_cv.R` — pointer to `forecast::tsCV`, `forecast::accuracy`, `fable::reconcile()`, `hts::MinT`.

## Assumptions

- **Time-ordered**: no look-ahead in fold construction.
- **MASE**: needs an in-sample naive baseline; seasonal `m > 1` for seasonal series.
- **Reconciliation**: hierarchy structure known.

## Run

```
python techniques/forecast-evaluation-cv/python/forecast_evaluation_cv.py
Rscript techniques/forecast-evaluation-cv/r/forecast_evaluation_cv.R
```

**Refs:** Hyndman, R.J. & Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed., OTexts, 2021 (Ch. 5); Hyndman, R.J. & Koehler, A.B. "Another look at measures of forecast accuracy." *IJF* 22(4), 679–688, 2006; Wickramasuriya, S.L., Athanasopoulos, G. & Hyndman, R.J. "Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization." *JASA* 114(526), 804–819, 2019; Makridakis, S., Spiliotis, E. & Assimakopoulos, V. "The M4 Competition: 100,000 time series and 61 forecasting methods." *IJF* 36(1), 54–74, 2020.

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
