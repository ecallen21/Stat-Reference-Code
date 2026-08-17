# Gradient Boosting (Reference §26.8)

Sequential additive model (Friedman 2001).

```
F_0(x) = mean(y)                            initial constant
for m = 1, ..., M:
    r_i = −∇ Loss(y_i, F_{m−1}(x_i))         pseudo-residuals (= y − F for L2)
    fit shallow tree h_m to (X, r)
    F_m = F_{m−1} + ν · h_m(x)               ν = learning rate (0.01 – 0.1)
```

Trees are **shallow** (depth 3–6) — small bias contribution each; the ensemble reduces bias through many additive updates. Shrinkage `ν` regularizes and typically needs 100–2000 trees at `ν = 0.01–0.1`.

## Files

- `python/gradient_boosting.py` — from-scratch L2 GBM built on the `decision-tree` module. Demo: test RMSE 0.353 vs sklearn `GradientBoostingRegressor` 0.350 with identical hyperparameters.
- `r/gradient_boosting.R` — `gbm::gbm(distribution = "gaussian")` (Ridgeway) or `xgboost::xgb.train` (Chen production).

## Loss functions

- **Regression**: L2 (squared) — implemented here; L1 / Huber for robustness.
- **Classification**: log-loss / binary deviance / multinomial deviance.
- **Custom**: any twice-differentiable loss works.

## Modern flavors

- **XGBoost** — second-order gradients (Newton), regularization, sparsity-aware splits, GPU support.
- **LightGBM** — histogram binning, leaf-wise growth; fastest.
- **CatBoost** — ordered boosting for categorical features + target encoding.

## Hyperparameters (tune roughly in this order)

- **learning_rate** × **n_trees** — trade one for the other; more trees at lower `ν` usually helps if you can afford compute.
- **max_depth** or **num_leaves** — 3–6 for GBM, 8–32 leaves for LightGBM.
- **min_samples_leaf / min_child_weight** — regularization.
- **subsample / colsample_bytree** — stochastic gradient boosting.

## When to use

- **Structured tabular** — GBM is usually the strongest single-model baseline on tabular data.
- **Kaggle-style competitions** — XGBoost / LightGBM / CatBoost dominate.
- **When you can afford a longer training run** than random forest.

## Cautions

- **Overfits** aggressively with too many trees + high learning rate; **use early stopping** on a held-out validation set.
- **Calibration** — boosted classifier probabilities are usually well-calibrated but check with a reliability diagram (see `calibration-scaling`).
- **Interpretability** — use SHAP / permutation importance / partial-dependence (see `feature-importance`).

## Run

```
python techniques/gradient-boosting/python/gradient_boosting.py
Rscript techniques/gradient-boosting/r/gradient_boosting.R
```

**Refs:** Friedman, J.H. "Greedy function approximation: a gradient boosting machine." *Ann. Stat.* 29(5), 1189–1232, 2001; Chen, T. & Guestrin, C. "XGBoost: a scalable tree boosting system." *KDD*, 2016; Ke, G. et al. "LightGBM: a highly efficient gradient boosting decision tree." *NIPS*, 2017.

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
