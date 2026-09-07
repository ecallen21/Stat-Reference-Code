# DeLong's Test for Comparing AUCs (Reference §21.3)

DeLong-DeLong-Clarke-Pearson (1988). Compare two AUCs computed on
the **same cases** (paired or independent) using Mann-Whitney U-
statistic structure theory. Gold-standard method behind
`pROC::roc.test`.

## Statistic

For observation-level **placement values**:

- `V_10(pos)` — for each positive, fraction of negatives it out-
  scores.
- `V_01(neg)` — for each negative, fraction of positives it under-
  scores.

Then `AUC = mean(V_10)`. Covariance of paired AUCs is computed from
the between-model covariance of the `V` vectors.

## When to use

- **Compare two ROC curves** on the same test set (paired).
- **Model-selection** benchmarks where lift over baseline matters.

## When NOT to use

- **Very small samples** — DeLong z is a large-sample
  approximation; use bootstrap for `n < 50`.
- **Non-ROC comparisons** — different discrimination measures need
  their own tests.

## Files

- `python/delong_auc_test.py` — placement-value computation +
  DeLong SE + z-test (custom). Demo (n=500, paired scores; A
  higher signal than B): **AUC_A 0.727, AUC_B 0.648, diff 0.079,
  DeLong SE 0.034, z 2.33, p 0.020**; sanity check on same-model
  returns p ≈ 1.
- `r/delong_auc_test.R` — `pROC::roc.test(method='delong')`,
  `riskRegression::Score`, `auc` (R); custom + `sklearn.metrics`
  (Python).

## Assumptions & caveats

- **Paired vs unpaired** — different SE formulas; make sure the
  covariance term is included when the same cases are scored by
  both models.
- **Ties** — mid-rank convention (0.5 weight) built into the
  placement definition.
- **Bootstrap alternative** — for small n or when the DeLong normal
  approximation is suspect.

## Related in this repo

- `discrimination-calibration`, `calibration-plots` — sibling
  metrics.
- `bootstrap-optimism-correction` — internal-validity workflow.
- `nri-idi` — reclassification-based comparisons.

## Run

```
python techniques/delong-auc-test/python/delong_auc_test.py
Rscript techniques/delong-auc-test/r/delong_auc_test.R
```

**Refs:** DeLong, E.R., DeLong, D.M., & Clarke-Pearson, D.L. "Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach." *Biometrics*, 1988.

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
