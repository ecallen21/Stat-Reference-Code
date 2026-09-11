# ADASYN Oversampling (Reference §47.248)

He, Bai, Garcia & Li (2008, IJCNN). Like SMOTE, but generates
MORE synthetic samples near HARD minority points (those with
more majority-class neighbours):

```
for each x_min:
    r_i = fraction of majority neighbours in k-NN
    g_i = r_i normalised over minority
Number of synths near x_min = round(g_i * (n_maj - n_min))
```

Focuses learning on the decision boundary. Where SMOTE
distributes synths uniformly across the minority class, ADASYN
adapts to LOCAL DIFFICULTY.

## Files

- `python/adasyn_oversampling.py` — density-adaptive from
  scratch with sklearn NN. Demo: n=2000, 20 features, 98/2
  imbalance. r-scores across minority: mean 0.86, max 1.00
  (all points are "hard"). Plain LR gets P=1.00, R=0.17,
  F1=0.29; after ADASYN synthesising 1274 samples,
  P=0.07, R=0.62, F1=0.13 — trades precision for a big recall
  gain concentrated at the hard boundary.
- `r/adasyn_oversampling.R` — `smotefamily::ADAS`,
  `imbalance::adas`, `UBL` (R); `imbalanced-learn.ADASYN`,
  from-scratch (Python).

## When to use

- **Hard-boundary minority** where SMOTE's uniform interpolation
  wastes synths on already-easy regions.
- **Batched imbalanced training** — pre-compute ADASYN samples
  once, then train.
- **Combined with cleaning** (ADASYN + Tomek / ENN).

## When NOT to use

- **Very noisy majority class near minority** — ADASYN pumps
  synths into noise; SMOTE-Tomek is safer.
- **Small minority class** — few neighbourhood signals for
  reliable `r_i`.
- **Categorical features** — plain ADASYN, like SMOTE, is
  continuous-only.

## Assumptions & caveats

- **`r_i` estimation** requires enough k-NN neighbours; small
  minorities give noisy densities.
- **Over-adaptive** — a single hard point can dominate synthetic
  count; consider capping `n_i`.
- **Same convexity assumption** as SMOTE — synths are convex
  combinations of minority points.
- **Fold isolation** — apply INSIDE cross-validation, never on
  the pooled dataset.

## Related in this repo

- `smote-oversampling` — the uniform-density baseline.
- `tomek-links-undersampling`, `edited-nn-cleaning` — boundary
  cleaning companions (ADASYN-Tomek).
- `focal-loss-imbalance`, `class-balanced-loss` — alternative
  cost-based routes.
- `class-imbalance` — the overall framing.

## Run

```
python techniques/adasyn-oversampling/python/adasyn_oversampling.py
Rscript techniques/adasyn-oversampling/r/adasyn_oversampling.R
```

**Refs:** He, H., Bai, Y., Garcia, E.A. and Li, S. "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning." In *IEEE IJCNN*, pp. 1322-1328, 2008.

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
