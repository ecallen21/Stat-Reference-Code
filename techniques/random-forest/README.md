# Random Forest (Reference §26.7)

Ensemble of decision trees (Breiman 2001).

1. Bootstrap-sample the training data for each tree.
2. At each split consider only a **random subset** of features (√p for classification, p/3 for regression).
3. Grow trees deep (no pruning).
4. Aggregate: mean (regression) or majority vote (classification).

## Why it works

Individual deep trees have **low bias but high variance**. Averaging cancels variance → forest has lower total error. Random feature-subset decorrelates trees, boosting the variance reduction.

## Out-of-bag (OOB) error

Each tree's bootstrap sample leaves out ~1/3 of rows. Predict each row using only the trees that didn't see it → OOB error, a cheap cross-validation substitute.

## Files

- `python/random_forest.py` — from-scratch RF built on the `decision-tree` module + OOB scoring. Demo: 3-blob classification training accuracy 98.3%, OOB accuracy 97.0% (matches sklearn 97.7% closely); regression OOB RMSE 0.36.
- `r/random_forest.R` — `randomForest::randomForest` (Liaw-Wiener) or `ranger::ranger` (fast).

## When to use

- **Tabular data** with mixed types — the default first-choice ML model.
- **Nonlinear boundaries** — trees find them automatically.
- **Feature importance** — mean decrease in impurity, or permutation-based (see `feature-importance`).

## When to prefer other methods

- **Gradient boosting** (`gradient-boosting`) usually wins on structured tabular problems; RF is simpler and needs less tuning.
- **Linear models** for high-dimensional sparse data or when interpretable coefficients matter.
- **Neural nets** for unstructured data (images, text, sequences).

## Hyperparameters

- **n_estimators**: more is better (variance ↓), until performance plateaus. 100–1000.
- **max_depth**: usually leave unbounded; regularize via `min_samples_leaf` instead.
- **mtry**: default √p / p/3 works well. Tune ±30% for marginal gains.
- **min_samples_leaf**: 1 (default) fine for classification, 5 for regression.

## Assumptions & caveats

- **Prediction variance** — RF underestimates prediction uncertainty; combine with conformal prediction for calibrated intervals (see `conformal-prediction`).
- **Class-imbalance** — use `class_weight` or SMOTE (see `class-imbalance`).
- **Correlated features** inflate individual mean-decrease-in-impurity importance; permutation importance is more reliable.

## Run

```
python techniques/random-forest/python/random_forest.py
Rscript techniques/random-forest/r/random_forest.R
```

**Refs:** Breiman, L. "Random forests." *Mach. Learn.* 45(1), 5–32, 2001; Hastie, T., Tibshirani, R. & Friedman, J. *The Elements of Statistical Learning*, 2nd ed., Springer, 2009 (Ch 15).

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
