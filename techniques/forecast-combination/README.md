# Forecast Combination (Reference §13.30)

Combining multiple forecasts often beats any single one — a phenomenon so robust it is known as the **forecast combination puzzle**: a naive equal-weighted average frequently outperforms elaborate optimally-fitted weights, especially out-of-sample. Intuition: uncorrelated forecast errors partially cancel, reducing variance.

## Methods

- **Simple average** — `f̂_t = (1/K) Σ_k f_{t,k}`. Robust, hard-to-beat baseline.
- **Trimmed average** — drop the top and bottom `α` fraction each period. Guards against occasional wildly-wrong forecasters.
- **Bates-Granger** (1969) — inverse-variance weights from training errors:

```
w_k ∝ 1 / Var(e_k)
```

- **Granger-Ramanathan** — OLS regression of the outcome on the forecasts (with or without the constraint that weights sum to 1 and stay non-negative).

- **Discounted / exponentially-weighted** — weight recent errors more heavily; useful under drift.

## When (and when not) to combine

- **Combine when** forecasters use different information / model classes; errors are decorrelated; individual accuracies are similar-ish.
- **Don't combine when** one forecaster dominates all others (equivalent to `w_1 = 1`); errors are perfectly correlated (no gain); weights are estimated on too little data (overfits the past errors).

## Files

- `python/forecast_combination.py` — simple / trimmed averages, Bates-Granger inverse-variance weights, and Granger-Ramanathan OLS (constrained and unconstrained). Demo: 5 noisy forecasters with per-model MSEs from 0.28 to 3.71; all combinations beat every individual (best combined MSE 0.13 vs best individual 0.28).
- `r/forecast_combination.R` — same three approaches in base R; `forecastHybrid::hybridModel` for a production pipeline.

## Assumptions & caveats

- **Training and test forecasts** must be aligned in time and correspond to the same target.
- **Estimated weights need enough data** — the "puzzle" is worst when training sample is small; simple mean wins.
- **Weights can drift** — refit periodically or use a rolling window.
- **Combining bad models** doesn't rescue them; ensure each forecaster is at least unbiased.

## Run

```
python techniques/forecast-combination/python/forecast_combination.py
Rscript techniques/forecast-combination/r/forecast_combination.R
```

**Refs:** Bates, J.M. & Granger, C.W.J. "The combination of forecasts." *Oper. Res. Q.* 20(4), 451–468, 1969; Granger, C.W.J. & Ramanathan, R. "Improved methods of combining forecasts." *J. Forecast.* 3(2), 197–204, 1984; Timmermann, A. "Forecast combinations." *Handbook of Economic Forecasting* 1, 135–196, 2006.

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
