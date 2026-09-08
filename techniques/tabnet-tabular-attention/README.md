# TabNet Tabular Attention (Reference §47.112)

Arik & Pfister (2021). Sequential decision architecture for tabular
data:

1. At each STEP an ATTENTIVE TRANSFORMER computes a SPARSE feature
   mask (sparsemax; Martins & Astudillo 2016) selecting which
   features to reason about.
2. A FEATURE TRANSFORMER processes the masked features.
3. Outputs across steps aggregate into the final prediction; masks
   aggregate into a per-instance FEATURE-IMPORTANCE map.

Competitive with GBDTs on many tabular benchmarks plus built-in
interpretability.

## Files

- `python/tabnet_tabular_attention.py` — very simplified TabNet
  with a linear feature-selector per step (sparsemax attention on
  correlations with residuals) + linear regression on masked
  features. Demo (n=800, p=10, truly active {0, 3, 7}):
  - Step 1 mask picks {0}, Step 2 picks {3}, Step 3 picks {7}
  - Final MSE 0.091 (baseline var y = 5.575)
  - Aggregate mask importance: **{0, 3, 7}** rank top-3.
- `r/tabnet_tabular_attention.R` — no first-class R port;
  `pytorch-tabnet`, `keras-tabnet` in Python.

## When to use

- **Interpretable tabular models** where step-wise attention is
  useful.
- **Sparse-feature situations** — the sparsemax mask does explicit
  feature selection.
- **When training GBDT is inconvenient** (e.g. streaming setting
  with categorical / sparse features).

## When NOT to use

- **When XGBoost / LightGBM / CatBoost win empirically** — usually
  the case on standard benchmarks.
- **Very small n** — TabNet needs enough data for step-wise attention.
- **When SHAP / EBM already give the interpretability you need**.

## Assumptions & caveats

- **Sparsemax** vs softmax matters — softmax gives dense
  attention.
- **# steps** trades interpretability (few) vs capacity (many).
- **Ghost batch normalisation** helps at scale; not shown here.
- **Auxiliary self-supervised head** (masked-feature-reconstruction)
  boosts performance in the paper.

## Related in this repo

- `attention-mechanism`, `transformer-encoder`,
  `transformer-decoder` — attention family.
- `xgboost-boosting`, `lightgbm-histogram-boosting`,
  `catboost-ordered-boosting`, `gradient-boosting`,
  `bart-bayesian-additive-regression-trees` — GBDT competitors.
- `explainable-boosting-machine`, `shap-values`,
  `shap-interactions`, `friedmans-h-statistic`, `pdp-ice-plots`,
  `ale-accumulated-local-effects` — interpretability cousins.
- `masked-language-modeling`, `byol-simsiam`,
  `contrastive-predictive-coding` — SSL cousins (TabNet has an
  MLM-style pretraining variant).

## Run

```
python techniques/tabnet-tabular-attention/python/tabnet_tabular_attention.py
Rscript techniques/tabnet-tabular-attention/r/tabnet_tabular_attention.R
```

**Refs:** Arik, S.Ö. & Pfister, T. "TabNet: Attentive interpretable tabular learning." *AAAI*, 2021; Martins, A.F.T. & Astudillo, R.F. "From softmax to sparsemax: A sparse model of attention." *ICML*, 2016.

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
