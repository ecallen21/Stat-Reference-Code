# Matthews Correlation Coefficient (Reference §47.253)

Matthews (1975). A BALANCED classification metric that stays
informative under class imbalance:

```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

Range `[-1, +1]`. +1 = perfect, 0 = random, −1 = perfect
inverse. Unlike accuracy or F1, MCC treats both classes
symmetrically and penalises trivial majority-only predictions.

## Files

- `python/matthews_correlation_coefficient.py` — Numpy MCC +
  sklearn parity check. Demo: n=10 000 with 1% positives.
  Trivial "all-negative" predictor gets accuracy 0.99, F1 0,
  MCC 0 (correctly exposes the trivial model). Random predictor
  gets accuracy 0.50, F1 ≈ 0.02, MCC ≈ 0. Perfect predictor
  gets 1.00 everywhere. Our MCC matches sklearn to 4 dp.
- `r/matthews_correlation_coefficient.R` — `mltools::mcc`,
  `yardstick::mcc`, `mccr::mccr`, `caret::confusionMatrix` (R);
  `sklearn.metrics.matthews_corrcoef`, from-scratch (Python).

## When to use

- **Imbalanced binary classification** — the modern
  recommendation (Chicco & Jurman 2020) over F1.
- **Detecting trivial baselines** — MCC = 0 for majority-only,
  unlike accuracy.
- **Multiclass** — MCC generalises via a Pearson correlation of
  one-hot predictions vs labels.

## When NOT to use

- **Skewed cost of FP vs FN** — MCC is symmetric; use
  cost-sensitive utility instead.
- **Ordinal labels** — MCC ignores ordering; use quadratic
  kappa or weighted metrics.
- **Continuous outputs uncalibrated** — MCC uses thresholded
  predictions; report AUC / Brier alongside.

## Assumptions & caveats

- **Zero denominator** — when any confusion-matrix row or
  column is empty, define MCC = 0 (our implementation and
  sklearn agree).
- **Multiclass MCC** exists but interpretation is subtler
  (correlation of one-hot vectors).
- **Threshold-dependent** — pair with a probability-based
  metric (AUROC, Brier) for full picture.
- **Small `n`** — MCC has non-negligible variance; bootstrap
  its confidence interval.

## Related in this repo

- `class-imbalance` — the overall framing.
- `discrimination-calibration` — separates discrimination
  from calibration.
- `cohens-kappa`, `weighted-kappa` — reliability cousins.
- `cross-entropy-log-loss`, `proper-scoring-rules-crps` —
  probability-based scores.

## Run

```
python techniques/matthews-correlation-coefficient/python/matthews_correlation_coefficient.py
Rscript techniques/matthews-correlation-coefficient/r/matthews_correlation_coefficient.R
```

**Refs:** Matthews, B.W. "Comparison of the Predicted and Observed Secondary Structure of T4 Phage Lysozyme." *Biochim. Biophys. Acta*, 405(2): 442-451, 1975; Chicco, D. and Jurman, G. "The advantages of the Matthews correlation coefficient over F1 score and accuracy." *BMC Genomics*, 21: 6, 2020.

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
