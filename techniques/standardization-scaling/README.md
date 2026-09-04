# Standardization, Centering & Scaling (Reference §41.4)

Gelman (2008), Enders-Tofighi (2007). Four common numeric-feature
scalings plus group-mean centering for multilevel models.

## Scalings

| Method | Formula | Use |
|---|---|---|
| **Z-score** | `(x − μ) / σ` | Most common; assumes non-crazy scale |
| **Min-max** | `(x − min) / (max − min)` | Bounded activations (NN) |
| **Robust** | `(x − median) / IQR` | Resistant to outliers |
| **Gelman /2SD** | `(x − μ) / (2σ)` | Coefficients comparable to binary predictors |

## Centering in multilevel models

- **Grand-mean centering** — `x − x̄_overall`; keeps between- and
  within-cluster variation confounded.
- **Group-mean centering** — `x − x̄_cluster`; separates level-1
  (within) from level-2 (between) effects.

## When to use

- **Regularised regression** (ridge, LASSO) — standardise or
  penalty is scale-dependent.
- **Distance-based models** (k-NN, k-means, SVM) — standardise or
  large-scale features dominate.
- **Multilevel models with contextual effects** — group-mean centre.

## When NOT to use

- **Tree-based models** — invariant to monotone transforms of any
  single feature; scaling is a no-op.
- **Interpretability required on original scale** — report in the
  original units.

## Files

- `python/standardization_scaling.py` — z-score, min-max, robust,
  Gelman /2SD + group-mean centering. Demo: outlier-heavy series
  shows how **robust scaling preserves the middle 50 %** while
  z-score is pulled by extremes; group-mean centering removes
  between-cluster differences leaving only within-cluster deviations.
- `r/standardization_scaling.R` — `base::scale`,
  `recipes::step_normalize`/`step_center`/`step_scale`/`step_range`
  (R); `sklearn.preprocessing.StandardScaler`/`MinMaxScaler`/
  `RobustScaler`/`MaxAbsScaler` (Python).

## Assumptions & caveats

- **Fit on train**, apply to test — never re-fit the scaler on the
  test set (data leakage).
- **Sparse features** — min-max destroys sparsity; use `MaxAbsScaler`.
- **After scaling** interpret coefficients in the scaled units.
- **Report the scaling** in the paper; different scalings → different
  numerical coefficients.

## Related in this repo

- `box-cox-transformation`, `yeo-johnson-transformation` —
  distribution-shape transforms.
- `standardized-coefficients` — comparing regression coefficient
  magnitudes.

## Run

```
python techniques/standardization-scaling/python/standardization_scaling.py
Rscript techniques/standardization-scaling/r/standardization_scaling.R
```

**Refs:** Gelman, A. "Scaling regression inputs by dividing by two standard deviations." *Statistics in Medicine*, 2008; Enders, C.K. & Tofighi, D. "Centering predictor variables in cross-sectional multilevel models: a new look at an old issue." *Psychological Methods*, 2007.

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
