# Prediction vs Confidence Intervals (Reference §39.14)

Steyerberg (2019 ch 15), Harrell (2015 ch 5). Two intervals for
regression that are often confused and always distinct.

| Interval | Question | Width |
|---|---|---|
| **Confidence** | Where is the **mean** response at `x*`? | Narrow |
| **Prediction** | Where will a **new individual** at `x*` land? | Wider (includes residual noise) |

## Formulae (simple linear regression)

```
CI at x*:  ŷ ± t · s · √(1/n + (x* − x̄)² / S_xx)
PI at x*:  ŷ ± t · s · √(1 + 1/n + (x* − x̄)² / S_xx)
```

The extra `1` under the PI square root is residual variance; it
dominates unless `n` is small or `x*` is far from `x̄`.

## When to use

- **CI**: reporting the average treatment effect / predicted mean.
- **PI**: quoting a range for a new patient's lab value, next
  month's sales, tomorrow's temperature.

## When NOT to use

- **Classification** — PIs are for continuous outcomes; use
  conformal prediction for classification confidence sets.
- **Nonlinear / ML models** — closed-form PIs fail. Use bootstrap,
  quantile regression, or conformal prediction.

## Files

- `python/prediction_intervals.py` — closed-form CI + PI for
  simple linear regression at four `x*` values, plus a coverage
  simulation. Demo (n=60, SBP-like `y ~ x` with σ=12): at x*=70,
  **CI width ≈ 6.4**, **PI width ≈ 49.7** (~8× wider); simulated
  95 % coverage — **CI covers true mean 0.952**, **PI covers new
  obs 0.947**.
- `r/prediction_intervals.R` — `stats::predict.lm`, `rms::Predict`,
  `ciTools::add_pi` (R); `statsmodels.wls_prediction_std` +
  custom (Python).

## Assumptions & caveats

- **Normal errors** — closed-form PI relies on it; heavy tails
  require bootstrap or quantile PIs.
- **Homoscedasticity** — heteroscedastic errors need
  variance-modelled PIs (`ciTools`, quantile regression).
- **Extrapolation** — PI (and CI) width grows as `(x* − x̄)²`;
  quoting either far outside the training range is speculative.
- **Model correctness** — a mis-specified mean structure inflates
  residuals and widens PI honestly but fails coverage if bias is
  strong.

## Related in this repo

- `conformal-classification` / `conformal-prediction` — non-
  parametric PIs with finite-sample coverage.
- `quantile-regression` — direct quantile-based PIs.
- `bootstrap-optimism-correction` — a different use of resampling
  for internal validity.

## Run

```
python techniques/prediction-intervals/python/prediction_intervals.py
Rscript techniques/prediction-intervals/r/prediction_intervals.R
```

**Refs:** Steyerberg, E.W. *Clinical Prediction Models*, 2nd ed., Springer, 2019 (ch 15); Harrell, F.E. *Regression Modeling Strategies*, 2nd ed., Springer, 2015 (ch 5).

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
