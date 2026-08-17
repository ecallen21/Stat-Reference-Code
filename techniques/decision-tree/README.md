# CART Decision Tree (Reference §26.6)

Recursive binary partitioning (Breiman-Friedman-Olshen-Stone 1984). At each node choose `(feature, threshold)` that best splits by an impurity criterion:

- **Regression**: variance reduction `ΔI = Var(parent) − (n_L/n) Var(L) − (n_R/n) Var(R)`.
- **Classification**: Gini `1 − Σ p_c²` or Entropy `−Σ p_c log p_c`.

Recurse until `max_depth`, `min_samples_leaf`, or negligible improvement. Predict: leaf mean (regression) or majority class (classification).

## Files

- `python/decision_tree.py` — from-scratch CART for both regression and classification with exhaustive threshold search. Demo: regression RMSE 0.26 on nonlinear target; classification accuracy 97.7% on 3-class (vs sklearn 99.7% with different hyperparameters).
- `r/decision_tree.R` — `rpart::rpart(y ~ ., data = df)` (Therneau-Atkinson canonical implementation).

## When to use

- **Interpretable model** — a small tree is directly readable as if-then rules.
- **Nonlinear boundaries** — trees capture axis-aligned decision surfaces natively.
- **Mixed data types** — numeric + categorical without preprocessing.
- **Feature interactions** — trees find them automatically.

## When NOT to use

- **Smooth target functions** — trees give piecewise-constant surfaces; splines or GAM are smoother.
- **High-dimensional data with linear structure** — regularized linear models (ridge/lasso) usually win.
- **When calibrated probabilities matter** — tree leaf frequencies are often over-confident; apply Platt or isotonic scaling (see `calibration-scaling`).

## Related methods

- **Random Forest** (`random-forest`) — bagged trees for variance reduction.
- **Gradient Boosting** (`gradient-boosting`) — sequential trees for bias reduction.
- **Extremely Randomized Trees**, **Rotation Forest** — variants.

## Assumptions & caveats

- **Greedy top-down splitting** finds local optima, not the globally-optimal tree.
- **Instability** — small data changes give very different trees. This is why ensembles work.
- **Bias toward high-cardinality features** — many possible thresholds → higher chance of a "good" split by chance. Regularize with `min_samples_split`.

## Run

```
python techniques/decision-tree/python/decision_tree.py
Rscript techniques/decision-tree/r/decision_tree.R
```

**Refs:** Breiman, L., Friedman, J.H., Olshen, R.A. & Stone, C.J. *Classification and Regression Trees*, Wadsworth, 1984; Hastie, T., Tibshirani, R. & Friedman, J. *The Elements of Statistical Learning*, 2nd ed., Springer, 2009 (Ch 9.2).

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
