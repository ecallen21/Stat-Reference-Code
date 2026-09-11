# SAINT — Row + Column Attention Transformer (Reference §47.241)

Somepalli, Goldblum, Schwarzschild, Bruss & Goldstein (2021).
Two attention flavours for tabular data:

1. **COL-ATTN**: attention across features within a row (like
   standard Transformer over tokens).
2. **ROW-ATTN**: attention across ROWS within a mini-batch — each
   row attends to peers with similar feature patterns.

Combined with contrastive pretraining (CutMix + MixUp on rows),
SAINT is a strong deep-tabular contender.

## Files

- `python/saint_tabular_transformer.py` — col-attn + row-attn
  forward pass on 50-row batches:
  - Cluster-geometry test: col-only ratio 8.30 vs col+row 8.40
    (row-attention marginally amplifies cluster structure with
    random weights; benefit compounds after trained).
- `r/saint_tabular_transformer.R` — no R port; recommends
  `somepalli/saint`, `tab-transformer-pytorch`.

## When to use

- **Tabular deep-learning benchmarks** competitive with GBM.
- **When cross-row context matters** (few-shot, small-batch
  transfer).
- **Combined with contrastive pretraining** on unlabelled tables.

## When NOT to use

- **When GBM (XGBoost / LightGBM) is enough** — often the case.
- **Very high-cardinality categoricals** — embedding table
  bottleneck.
- **Small datasets** — SAINT needs ~10k+ rows.

## Assumptions & caveats

- **Row-attention scales as O(N²)** in batch size — cap N.
- **CutMix / MixUp augmentation** essential for pretraining.
- **Feature tokeniser** (numerical + categorical) — same as
  FT-Transformer.

## Related in this repo

- `ft-transformer-tabular`, `tabnet-tabular-attention` — sibling
  tabular architectures.
- `xgboost-boosting`, `lightgbm-histogram-boosting`,
  `catboost-ordered-boosting` — GBM baselines.
- `mae-masked-autoencoders`,
  `simclr-contrastive` — SSL cousins.

## Run

```
python techniques/saint-tabular-transformer/python/saint_tabular_transformer.py
Rscript techniques/saint-tabular-transformer/r/saint_tabular_transformer.R
```

**Refs:** Somepalli, G. et al. "SAINT: Improved neural networks for tabular data via row attention and contrastive pre-training." *arXiv:2106.01342*, 2021.

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
