# Class Imbalance: SMOTE + Weighting + Threshold Tuning (Reference §26.17)

**Rare-event classification** — minority class ~ 1–5%. Vanilla accuracy is useless (99% by predicting all majority); use precision, recall, F1, PR-AUC, expected cost.

## Three complementary tactics

### 1. Class weighting

Weight minority examples in the loss by `n_maj / n_min`. Trivially available in most classifiers (`class_weight = "balanced"`).

### 2. Resampling

- **Random oversampling** — duplicate minority rows. Overfitting risk.
- **Random undersampling** — drop majority rows. Loses information.
- **SMOTE (Chawla et al. 2002)** — synthesize new minority points:

```
x_new = x_i + rand · (x_i_neighbor − x_i)         x_i_neighbor is a k-NN minority
```

More robust than random duplication for numeric features. Variants: Borderline-SMOTE, ADASYN, SMOTENC (mixed types).

### 3. Threshold tuning

Default 0.5 threshold optimizes accuracy; **tune the score cutoff** on a held-out set for F1 / precision-at-recall / expected cost.

## Files

- `python/class_imbalance.py` — from-scratch SMOTE + class-weighted logistic + threshold tuner. Demo (n = 1000, 4.5% positive):
  - Naive logistic: precision 0.87, recall 0.76, F1 0.81
  - Class-weighted: precision 0.46, recall 0.98, F1 0.63 (recall-heavy)
  - SMOTE + naive: precision 0.55, recall 0.91, F1 0.69
  - Threshold-tuned naive at 0.35: precision 0.80, recall 0.82, F1 0.81 (best F1)

- `r/class_imbalance.R` — `DMwR2::SMOTE`, `ROSE::ovun.sample`, `themis::` recipe steps.

## When to use which

- **Weighting** — cheapest; try first with any classifier that supports it.
- **SMOTE** — when weighting isn't enough and features are numeric.
- **Threshold tuning** — always the last step; tune to the operational cost curve.
- **Cost-sensitive learning** — encode misclassification costs directly.

## Metrics for imbalanced classification

- **PR-AUC (average precision)** — replaces AUC when positives are rare.
- **F1 / F_beta** — beta > 1 weights recall more.
- **Cohen's kappa / MCC** — chance-corrected agreement.
- **Precision @ K** — top-K ranking metric.
- **Expected cost** — `c_FP · FP + c_FN · FN`.

## Assumptions & caveats

- **SMOTE for text or high-D features** doesn't work well — synthesized points fall off the data manifold. Use SMOTE-NC or don't resample.
- **Resample only the training set** — never SMOTE the test data.
- **Cross-validation** — resample inside each fold, not before splitting.

## Run

```
python techniques/class-imbalance/python/class_imbalance.py
Rscript techniques/class-imbalance/r/class_imbalance.R
```

**Refs:** Chawla, N.V. et al. "SMOTE: Synthetic minority over-sampling technique." *JAIR* 16, 321–357, 2002; He, H. & Garcia, E.A. "Learning from imbalanced data." *IEEE TKDE* 21(9), 1263–1284, 2009.

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
