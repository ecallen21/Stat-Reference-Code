# FT-Transformer — Feature-Tokeniser Transformer (Reference §47.242)

Gorishniy, Rubachev, Khrulkov & Babenko (2021, NeurIPS). Two ideas:

1. **Feature Tokeniser**: each feature (numerical / categorical)
   gets its own d-dim embedding. Numerical feature k is embedded
   as v_k = x_k · w_k + b_k; categorical as an ordinary embedding
   lookup.
2. Standard Transformer encoder + **[CLS] token** for prediction.

Beats vanilla MLPs / TabTransformer / TabNet on most tabular
benchmarks (Gorishniy 2021 comparison).

## Files

- `python/ft_transformer_tabular.py` — random-init attention +
  LS-fit [CLS] head on a synthetic linear-regression task. The
  demo illustrates the tokeniser + [CLS] pipeline; a random-init
  attention layer is expected to underperform linear regression
  on truly linear data.
- `r/ft_transformer_tabular.R` — no R port; recommends
  `yandex-research/rtdl`, `pytorch-tabular`, `tabnet`.

## When to use

- **Tabular deep-learning research** benchmarks.
- **When both numerical and categorical features exist** and a
  unified attention token space helps.
- **Combined with SSL pretraining** on unlabelled tables.

## When NOT to use

- **When GBM (XGBoost / LightGBM) is enough** — usually the case
  in production.
- **Small datasets** (< 1k rows) — attention will overfit.

## Assumptions & caveats

- **Numerical embedding** — most common: linear-plus-bias.
- **Categorical embedding tables** — standard dim ~ 32-128.
- **[CLS] readout** vs pooled-features — [CLS] is the paper
  default.
- **Position embedding not needed** — features are unordered.

## Related in this repo

- `tabnet-tabular-attention`,
  `saint-tabular-transformer` — sibling tabular architectures.
- `xgboost-boosting`, `lightgbm-histogram-boosting`,
  `catboost-ordered-boosting` — GBM baselines.
- `vision-transformer-vit` — CLS-token ancestor.

## Run

```
python techniques/ft-transformer-tabular/python/ft_transformer_tabular.py
Rscript techniques/ft-transformer-tabular/r/ft_transformer_tabular.R
```

**Refs:** Gorishniy, Y., Rubachev, I., Khrulkov, V. & Babenko, A. "Revisiting deep learning models for tabular data (FT-Transformer)." *NeurIPS*, 2021.

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
