# Cross-Validation: K-fold, Stratified, LOOCV (Reference §10.8; also covers §10.12)

Estimate the **out-of-sample performance** of a model by partitioning the data, training on one part, testing on the held-out part, and averaging.

| Scheme | How it splits | When to use |
|---|---|---|
| **K-fold** | Partition into K folds; train on K−1, test on 1 | General default; K = 5 or 10 |
| **Stratified K-fold** | Same as K-fold but preserves class proportions | Classification, especially imbalanced |
| **LOOCV** | K = n; each observation is a test fold | Small n (< ~50); expensive elsewhere |

## Bias/variance trade-off in K

- **Small K** (2, 5): more bias (training set much smaller than n), less variance.
- **Large K** / LOOCV: less bias, more variance (fold predictions are highly correlated because training sets overlap so much).
- **K = 5 or 10** is the usual sweet spot.

## Files

- `python/cross_validation.py` — from-scratch `kfold_indices` / `stratified_kfold_indices` / `loocv_indices` + a generic `cv_score(X, y, fit_fn, predict_fn, score_fn, splitter, ...)` driver + OLS convenience. Mean MSE matches `sklearn.model_selection.cross_val_score` closely.
- `r/cross_validation.R` — from-scratch splitters + `cv_score`.
- `pyspark/cross_validation.py` — MLlib `CrossValidator` + `ParamGridBuilder` on a linear-regression pipeline; picks best `regParam` from the grid.

## Assumptions

- **Independent observations**. For time-series, use forward-chaining / rolling-origin CV instead.
- The **splitter** must not leak information (e.g. stratify by group ID, not by row, when observations cluster by subject).
- Nested CV (see [`nested-cv`](../nested-cv)) when you also tune hyperparameters, or the CV MSE becomes optimistically biased.

## Run

```
python techniques/cross-validation/python/cross_validation.py
Rscript techniques/cross-validation/r/cross_validation.R
python techniques/cross-validation/pyspark/cross_validation.py
```

**Refs:** Stone, M. "Cross-validatory choice and assessment of statistical predictions." *JRSS B* 36(2), 111–147, 1974; Geisser, S. "The predictive sample reuse method with applications." *JASA* 70(350), 320–328, 1975; Kohavi, R. "A study of cross-validation and bootstrap for accuracy estimation and model selection." *IJCAI*, 1995; Hastie, T., Tibshirani, R. & Friedman, J. *The Elements of Statistical Learning*, 2nd ed., Springer, 2009 (Ch. 7).

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
