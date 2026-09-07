# Youden J / ROC01 Optimal Cutpoint (Reference §26.10)

Youden (1950). Choose the threshold `t` that maximises

    J(t) = Sensitivity(t) + Specificity(t) − 1
         = TPR(t) − FPR(t)

Geometrically, the ROC point farthest from the chance diagonal.

## Alternatives

| Criterion | Objective |
|---|---|
| Youden J | max `Sen + Spe − 1` |
| ROC01 | min `(1 − Sen)² + (1 − Spe)²` (closest to (0, 1)) |
| Cost-weighted | min `c_FN · (1 − Sen) · prev + c_FP · (1 − Spe) · (1 − prev)` |
| MCC | max Matthews correlation coefficient |

## Files

- `python/youden_optimal_cutpoint.py` — Youden and ROC01 sweeps
  from scratch. Demo (n=2000, 25% prevalence): both criteria pick
  t* = 0.680 with Sen 0.744 / Spe 0.780; naive t=0.5 gives Sen
  0.912 / Spe 0.509 (over-sensitive, under-specific).
- `r/youden_optimal_cutpoint.R` — `cutpointr`, `OptimalCutpoints`,
  `pROC::coords` (R); `sklearn.metrics.roc_curve` + argmax, from-
  scratch (Python).

## When to use

- **Diagnostic test cutoff selection** — screening vs confirmation.
- **Equal weighting of Sen and Spe** — Youden implicitly weights
  them equally.
- **Reporting sensitivity + specificity together** at one operating
  point (clinical guideline).

## When NOT to use

- **Cost imbalance** — FN and FP consequences differ; use cost-
  weighted or expected-utility.
- **Skewed prevalence** — F1 threshold or PR-curve optimum may be
  more appropriate.
- **Continuous decisions** — dichotomising loses information;
  report scores where possible.

## Assumptions & caveats

- **Held-out data** — pick t on validation, not training.
- **Prevalence-dependent** — the ROC01 cutpoint is prevalence-
  invariant; Youden depends on prevalence via the finite-sample
  ROC.
- **Report CIs** — Perkins & Schisterman 2006 warn that "optimal"
  cutpoints are unstable; bootstrap CI on t*.
- **Continuous refinement** — grid search then a fine local
  search around the max.

## Related in this repo

- `f1-optimal-threshold` — F-β analogue.
- `roc-auc-analysis`, `delong-auc-test`,
  `calibration-scaling`, `discrimination-calibration` — classifier
  evaluation cousins.
- `decision-curve-analysis`, `cost-effectiveness-analysis` —
  utility-based decision tools.

## Run

```
python techniques/youden-optimal-cutpoint/python/youden_optimal_cutpoint.py
Rscript techniques/youden-optimal-cutpoint/r/youden_optimal_cutpoint.R
```

**Refs:** Youden, W.J. "Index for rating diagnostic tests." *Cancer*, 3(1): 32-35, 1950; Perkins, N.J. & Schisterman, E.F. "The inconsistency of `optimal' cutpoints obtained using two criteria based on the receiver operating characteristic curve." *American Journal of Epidemiology*, 163(7): 670-675, 2006.

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
