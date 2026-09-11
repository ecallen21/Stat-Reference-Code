# SMOTE Oversampling (Reference §47.247)

Chawla, Bowyer, Hall & Kegelmeyer (2002). Generate synthetic
minority samples by interpolating between a minority example
and one of its `k` nearest minority neighbours:

```
for x_min in minority:
    pick a nearest minority neighbour x_nn
    alpha ~ Uniform(0, 1)
    new = x_min + alpha * (x_nn - x_min)
```

Balances rare-class classification without duplicating exact
copies (which cause overfitting).

## Files

- `python/smote_oversampling.py` — Numpy + sklearn NN. Demo:
  n=2000, 20 features, 98/2 imbalance, 5% label flip. Plain
  LR gets P=1.000, R=0.167, F1=0.286; after SMOTE oversampling
  to 1:1 balance, LR shifts to P=0.086, R=0.667, F1=0.152.
  Recall LIFTS from 0.17 → 0.67 at the cost of precision — the
  characteristic SMOTE trade.
- `r/smote_oversampling.R` — `smotefamily`, `DMwR`,
  `performanceEstimation`, `UBL` (R); `imbalanced-learn.SMOTE`
  / `SMOTENC` / `SMOTEN`, from-scratch (Python).

## When to use

- **Rare-class classification** where recall / catching
  positives matters (fraud, disease screening).
- **Continuous features** — interpolation is well-defined.
- **Combined with cleaning** (SMOTE-Tomek, SMOTE-ENN) for a
  balanced F1 improvement.

## When NOT to use

- **Categorical-only features** — plain SMOTE produces
  invalid interpolations; use SMOTE-N (nominal) or SMOTENC
  (mixed).
- **Very small minority class** (e.g. < 6) — poor neighbour
  quality; try random oversampling, cost-sensitive learning,
  or focal loss.
- **Precision matters more than recall** — SMOTE lowers
  precision; consider class-weighted models instead.
- **Test-set contamination risk** — apply SMOTE INSIDE the
  cross-validation fold, never before splitting.

## Assumptions & caveats

- **Local convexity** — assumes the minority class is
  approximately convex within k-NN neighbourhoods; violated by
  disjoint sub-clusters (BorderlineSMOTE / KMeansSMOTE help).
- **Overlapping regions** — SMOTE will place synths in the
  majority zone; combine with Tomek / ENN cleaning.
- **Curse of dimensionality** — NN distances become unreliable
  in >50 features; project first (e.g. PCA) or use ADASYN.
- **Assumes IID minority** — for grouped data (patients,
  households), use group-aware resampling to avoid leakage.

## Related in this repo

- `adasyn-oversampling` — density-adaptive SMOTE variant.
- `tomek-links-undersampling`, `edited-nn-cleaning` —
  boundary-cleaning companions.
- `focal-loss-imbalance`, `class-balanced-loss` — cost-based
  alternatives without resampling.
- `class-imbalance` — the overall problem framing.
- `matthews-correlation-coefficient` — better metric for
  imbalanced eval.

## Run

```
python techniques/smote-oversampling/python/smote_oversampling.py
Rscript techniques/smote-oversampling/r/smote_oversampling.R
```

**Refs:** Chawla, N.V., Bowyer, K.W., Hall, L.O. and Kegelmeyer, W.P. "SMOTE: Synthetic Minority Over-sampling Technique." *J. Artif. Intell. Res.*, 16: 321-357, 2002.

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
