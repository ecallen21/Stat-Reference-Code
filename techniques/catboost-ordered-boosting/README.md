# CatBoost / Ordered Boosting (Reference §47.88)

Prokhorenkova, Gusev, Vorobev, Dorogush & Gulin (2018). Two
innovations over XGBoost / LightGBM:

1. **Ordered target statistics** — for each row `i`, encode a
   category using only rows `j < i` in a random permutation with
   matching category. Removes target-leakage bias present in
   plain target-mean encoding.
2. **Ordered boosting** — gradient estimates for row `i` come from
   a model trained WITHOUT row `i`, reducing the "prediction
   shift" bias found in classical GBM.

## Files

- `python/catboost_ordered_boosting.py` — from-scratch demo of
  the ordered-target-statistic trick vs plain leaky target
  encoding, fed into `sklearn.GradientBoostingRegressor`. Demo
  (n=800, 30-level categorical + continuous feature):
  - **In-sample train MSE**: leaky enc 0.066 vs ordered 0.134
    (leakage lets the model over-fit the target it "saw")
  - **5-fold CV MSE**: leaky 0.115 vs ordered 0.138 — leakage
    inflates apparent CV performance too, because the encoder
    uses the (train-fold) target.
- `r/catboost_ordered_boosting.R` — `catboost` (R + Python
  binary); `catboost` and `sklearn` alternatives (Python).

## When to use

- **High-cardinality categoricals** — job title, product SKU,
  city, gene ID.
- **When leakage is suspected** in plain target encoding
  pipelines.
- **Tabular data at scale** — CatBoost consistently competitive
  with LightGBM / XGBoost on Kaggle-style tabular.
- **GPU training** — CatBoost supports GPU trees efficiently.

## When NOT to use

- **Very small datasets** — LR / RF simpler and adequate.
- **Deep-feature interactions in vision / language** — DL better
  suited.
- **When categorical count is tiny** (< 20) — one-hot is fine;
  ordered target statistics add complexity without benefit.

## Assumptions & caveats

- **Permutation choice matters** — CatBoost averages over several
  permutations at build time (this demo uses one for simplicity).
- **Time-series** — ordered target statistics can respect temporal
  order if permutation ≡ time.
- **Prior on target mean** for low-count categories — CatBoost adds
  a Bayesian smoothing term (not shown here).
- **Symmetric (oblivious) trees** are CatBoost's default; more
  regularised than XGBoost's typical asymmetric trees.

## Related in this repo

- `gradient-boosting`, `bagging-oob`, `adaboost-classifier`,
  `random-forest`, `bart-bayesian-additive-regression-trees`,
  `explainable-boosting-machine` — tree-ensemble family.
- `target-encoding`, `feature-hashing`,
  `categorical-variable-coding` — encoding schemes CatBoost
  competes with.
- `hte-uplift`, `causal-forest`, `dml-double-ml` — HTE / causal
  cousins.
- `shap-values`, `shap-interactions`, `pdp-ice-plots`,
  `ale-accumulated-local-effects` — post-hoc interpretation of
  tree ensembles.

## Run

```
python techniques/catboost-ordered-boosting/python/catboost_ordered_boosting.py
Rscript techniques/catboost-ordered-boosting/r/catboost_ordered_boosting.R
```

**Refs:** Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V. & Gulin, A. "CatBoost: unbiased boosting with categorical features." *NeurIPS*, 2018.

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
