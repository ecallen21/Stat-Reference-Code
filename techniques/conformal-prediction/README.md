# Conformal Prediction (Reference §10.19)

**Distribution-free prediction intervals with finite-sample coverage guarantee.** Given just exchangeability of `(X_i, y_i)`, the split-conformal interval covers the true `y_new` with probability at least `1 − α`. Works with **any** underlying model as a black box (linear regression, random forest, XGBoost, neural network).

## Split-conformal (Papadopoulos et al. 2002; Lei-Wasserman 2014)

```
1. Split data into TRAIN and CALIBRATION sets.
2. Fit model on TRAIN → μ̂.
3. Nonconformity scores on CALIBRATION:  s_i = |y_i − μ̂(x_i)|.
4. q = ⌈(n_cal + 1)(1 − α)⌉ / n_cal quantile of {s_i}.
5. Interval:  μ̂(x_new) ± q.
```

## Variants

- **Locally-adaptive** (Lei et al. 2018): `s_i = |y_i − μ̂(x_i)| / σ̂(x_i)` with a variance model. Interval width varies with `x`.
- **CQR — Conformalized Quantile Regression** (Romano-Patterson-Candès 2019): fit quantile regressors `q̂_lo, q̂_hi` and score `s_i = max(q̂_lo(x_i) − y_i, y_i − q̂_hi(x_i))`. Distribution-free intervals **with** heteroscedasticity adaptivity.
- **Full conformal** — no data split; refit for each new prediction. Expensive.
- **Jackknife+, CV+** (Barber et al. 2021) — average calibration over multiple folds.

## Files

- `python/conformal_prediction.py` — split-conformal regression with any user-supplied `fit_predict` (defaults to OLS). Demo (n = 500, target coverage 90%): empirical coverage 93% on held-out; interval width 3.48 for a Normal-noise problem.
- `r/conformal_prediction.R` — base R split-conformal helper; production `conformalInference` package.

## When to use

- **Any predictive model** where you want calibrated uncertainty without distributional assumptions.
- **Post-hoc calibration** of a black-box ML model.
- **Regression, classification, quantile prediction, time series** — the framework generalizes.

## Guarantees & caveats

- **Marginal** coverage over the joint distribution of `(X, y)` — not conditional coverage `Pr(y ∈ interval | X)`.
- **Exchangeability required** — i.i.d. is sufficient; time-series settings need adaptations (e.g. weighted conformal, AdaptiveConformalInference).
- **Efficiency depends on the model**: bad `μ̂` → wide intervals but still calibrated. Better `μ̂` → tighter intervals.
- **Split cost**: half the data goes to calibration; use CV+ or jackknife+ for full-data efficiency.

## Run

```
python techniques/conformal-prediction/python/conformal_prediction.py
Rscript techniques/conformal-prediction/r/conformal_prediction.R
```

**Refs:** Vovk, V., Gammerman, A. & Shafer, G. *Algorithmic Learning in a Random World*, Springer, 2005; Papadopoulos, H. et al. "Inductive confidence machines for regression." *ECML*, 2002; Lei, J. et al. "Distribution-free predictive inference for regression." *JASA* 113(523), 1094–1111, 2018; Angelopoulos, A.N. & Bates, S. "A gentle introduction to conformal prediction and distribution-free uncertainty quantification." arXiv:2107.07511, 2021.

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
