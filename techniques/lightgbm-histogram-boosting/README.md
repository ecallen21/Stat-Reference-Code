# LightGBM / Histogram GBM (Reference §47.105)

Ke, Meng, Finley, Wang, Chen, Ma, Ye & Liu (2017). Contrast with
XGBoost:

- **Histogram** split finding: pre-bin features to 256 values;
  O(n·d) preprocessing then O(bins·d) per split.
- **Leaf-wise** (best-first) tree growth vs level-wise.
- **GOSS** (Gradient One-Side Sampling): keep large-|g| rows.
- **EFB** (Exclusive Feature Bundling): bundle sparse features.

Order-of-magnitude faster than XGBoost on many datasets at
comparable or better accuracy.

## Files

- `python/lightgbm_histogram_boosting.py` — benchmark of
  `sklearn.GradientBoostingRegressor` (exact splits) vs
  `sklearn.ensemble.HistGradientBoostingRegressor` (histogram
  variant that shares the LightGBM idea) + native LightGBM when
  installed. Demo (n=20 000, 20 features):
  - Exact-split GBM: 23.5 s, test MSE 0.343
  - Histogram GBM:    0.36 s, test MSE 0.293 (**65× speedup**).
- `r/lightgbm_histogram_boosting.R` — `lightgbm` (R + Python)
  reference; `xgboost`, `sklearn.HistGradientBoosting` alternatives.

## When to use

- **Large tabular data** (n > 10⁵) — LightGBM's core niche.
- **Wide-and-sparse feature spaces** — EFB shrinks work.
- **Fast iteration during hyperparameter search** — histogram
  splits are cheap.
- **Ranking / classification / regression** with rich features.

## When NOT to use

- **Very small n** — GOSS / EFB do not help; use simpler GBM.
- **When native categorical support isn't enough** — CatBoost's
  ordered TS may be needed for high-cardinality vars.
- **When strict monotonic constraints matter** — LightGBM supports
  them but XGBoost's constraints are more mature.

## Assumptions & caveats

- **Leaf-wise growth** can overfit; regularise with `num_leaves`,
  `max_depth`, `min_data_in_leaf`.
- **Histogram binning** — 256 bins standard; higher = less bias,
  more memory.
- **GPU LightGBM** faster than CPU but early-2020s support less
  mature than XGBoost's GPU histogram.
- **Determinism** — parallel splits can produce slightly different
  trees across runs; fix `deterministic=True`.

## Related in this repo

- `xgboost-boosting`, `catboost-ordered-boosting`,
  `gradient-boosting`, `adaboost-classifier`,
  `random-forest`, `bart-bayesian-additive-regression-trees`,
  `bagging-oob` — tree-ensemble family.
- `shap-values`, `shap-interactions`, `friedmans-h-statistic`,
  `pdp-ice-plots`, `ale-accumulated-local-effects` — XAI companions.
- `target-encoding`, `feature-hashing` — feature-engineering
  neighbours.

## Run

```
python techniques/lightgbm-histogram-boosting/python/lightgbm_histogram_boosting.py
Rscript techniques/lightgbm-histogram-boosting/r/lightgbm_histogram_boosting.R
```

**Refs:** Ke, G. et al. "LightGBM: A highly efficient gradient boosting decision tree." *NeurIPS*, 2017.

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
