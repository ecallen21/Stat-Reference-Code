# XGBoost (Reference §47.104)

Chen & Guestrin (2016). Gradient boosting with:

- **Second-order (Newton) Taylor** expansion of the loss.
- **Regularised** objective: `½λ‖w‖² + γT` (T = # leaves).
- **Sparse-aware** split finding + histogram / approximate splits.
- **Column-block caching** and parallel implementations.

Closed-form optimal leaf weight `w* = −G / (H + λ)` and split gain
`½ [G_L² / (H_L+λ) + G_R² / (H_R+λ) − G_S² / (H_S+λ)] − γ`.

## Files

- `python/xgboost_boosting.py` — from-scratch Newton boosting
  for squared loss (sklearn `DecisionTreeRegressor` as base learner)
  with the regularised leaf-weight update, plus a benchmark against
  `xgboost.XGBRegressor` when installed. Demo (n=800, p=8, mixed
  nonlinear signal):
  - T=50  test MSE = 1.31
  - T=200 test MSE = 1.05
  - T=500 test MSE = **0.82**.
- `r/xgboost_boosting.R` — `xgboost` (R + Python) reference;
  `lightgbm`, `catboost`, `sklearn.GradientBoostingRegressor`
  alternatives.

## When to use

- **Tabular data** — XGBoost has been the standard Kaggle winner
  since 2015.
- **Ranking, classification, regression** with rich structured
  features.
- **Very large n, moderate d** — histogram + parallel splits.
- **Custom loss functions** — as long as gradient + Hessian are
  supplied.

## When NOT to use

- **Small-to-moderate data with truly linear signal** — regularised
  linear model wins.
- **Deep-feature pipelines** — DL beats trees for images / text.
- **When you need calibrated probabilities out of the box** — trees
  need Platt / isotonic recalibration.

## Assumptions & caveats

- **Overfitting** — regularise via `max_depth`, `min_child_weight`,
  `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha`.
- **Hyperparameter tuning** — Bayesian / random search; grid is
  wasteful.
- **Feature importance** — split gain / cover / weight; SHAP for
  local attributions.
- **Categorical features** — historically require one-hot / target
  encoding; XGBoost 1.6+ has native categorical support.

## Related in this repo

- `gradient-boosting`, `adaboost-classifier`,
  `lightgbm-histogram-boosting`,
  `catboost-ordered-boosting`, `bart-bayesian-additive-regression-trees`,
  `explainable-boosting-machine`, `bagging-oob`, `random-forest` —
  tree-ensemble family.
- `shap-values`, `shap-interactions`,
  `friedmans-h-statistic`, `pdp-ice-plots` — XAI companions.
- `coordinate-descent-lasso`, `admm-consensus`,
  `ridge-lasso-elasticnet` — regularised alternatives.

## Run

```
python techniques/xgboost-boosting/python/xgboost_boosting.py
Rscript techniques/xgboost-boosting/r/xgboost_boosting.R
```

**Refs:** Chen, T. & Guestrin, C. "XGBoost: A scalable tree boosting system." *KDD*, 2016.

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
