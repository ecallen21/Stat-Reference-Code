# ROC + AUC Analysis (Reference §21.5)

Binary-classifier / diagnostic-test evaluation.

## ROC curve

For every possible threshold `t` on a continuous score `s(x)`:

```
TPR(t)  = Pr(s > t | positive)     sensitivity
FPR(t)  = Pr(s > t | negative)     1 − specificity
```

Plot `TPR` vs `FPR` as `t` varies.

## AUC

```
AUC = ∫_0^1 TPR(FPR) d(FPR) = Pr(s(X_+) > s(X_−))
```

- `AUC = 0.5` — random.
- `AUC = 1.0` — perfect ranking.
- Equal to a normalized **Mann-Whitney U** statistic.

## SE and CI

- **DeLong et al. (1988)** — analytical SE from the Mann-Whitney U variance; standard in medical stats.
- **Hanley-McNeil (1982)** — parametric approximation, adequate for AUC not too close to 0.5 or 1.
- **Bootstrap** — always available; use for very small samples or comparing correlated AUCs.

## Operating-point summaries

- **Youden's J** = `TPR − FPR`; threshold maximizing `J` is a common "optimal" cut.
- **Sensitivity at fixed specificity** — clinically common.
- **Partial AUC** — restrict integration to a high-specificity region (`sp ≥ 0.9`) when only that region matters.

## Files

- `python/roc_auc_analysis.py` — ROC curve + Mann-Whitney AUC + Hanley-McNeil SE + Youden J + normalized partial AUC. Demo (n_pos = 100, n_neg = 200, class shift +1σ): AUC = 0.7951 matches `sklearn.metrics.roc_auc_score` exactly; Youden J = 0.47; partial AUC (sp ≥ 0.8) normalized = 0.32.
- `r/roc_auc_analysis.R` — `pROC::roc` with `pROC::ci.auc(method = "delong")` for the canonical R implementation with DeLong CI and paired-AUC test.

## When to use

- **Any binary predictor / test** where the score is continuous and the threshold isn't fixed.
- **Model comparison** on the same holdout: report ΔAUC + DeLong z-test.
- **Reporting** classifier performance in medical / risk / recommendation applications.

## Cautions

- **AUC does not depend on class prevalence** — same AUC across different base rates. Use **precision-recall (PR)** curves for very imbalanced classes.
- **Calibration** is orthogonal to discrimination — a perfectly-discriminating classifier can still be miscalibrated.
- **Do not derive the "optimal" threshold from AUC alone** — depends on the misclassification cost ratio.
- **Comparing two AUCs on the same subjects**: use paired DeLong (`pROC::roc.test(r1, r2, paired = TRUE)`).

## Run

```
python techniques/roc-auc-analysis/python/roc_auc_analysis.py
Rscript techniques/roc-auc-analysis/r/roc_auc_analysis.R
```

**Refs:** Hanley, J.A. & McNeil, B.J. "The meaning and use of the area under a receiver operating characteristic (ROC) curve." *Radiology* 143(1), 29–36, 1982; DeLong, E.R., DeLong, D.M. & Clarke-Pearson, D.L. "Comparing the areas under two or more correlated receiver operating characteristic curves." *Biometrics* 44(3), 837–845, 1988; Pepe, M.S. *The Statistical Evaluation of Medical Tests for Classification and Prediction*, OUP, 2003.

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
