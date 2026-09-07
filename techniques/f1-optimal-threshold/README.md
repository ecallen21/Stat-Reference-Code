# F-β Optimal Decision Threshold (Reference §26.9)

For a probabilistic binary classifier `p ∈ [0, 1]`, the naive rule
`p > 0.5` is Bayes-optimal only for balanced classes. Sweep `t` to
maximise `F_β`:

    F_β = (1 + β²) · P · R / (β² · P + R)

β < 1 favours precision; β > 1 favours recall.

## Files

- `python/f1_optimal_threshold.py` — F-β sweep from scratch. Demo
  (n=3000, 5% positive class): t=0.5 gives F1 = 0.32; the F1-
  optimal t* = 0.765 lifts F1 to 0.49 (57% improvement). F0.5 and
  F2 each pick different thresholds matching their precision /
  recall preference.
- `r/f1_optimal_threshold.R` — `yardstick::f_meas`,
  `probably::threshold_perf` (R); `sklearn.metrics.fbeta_score`,
  `sklearn.metrics.precision_recall_curve` (Python).

## When to use

- **Imbalanced classification** — F1 threshold ≠ 0.5.
- **Business metric maps to precision / recall trade-off** — e.g.
  fraud (favour recall), spam (favour precision).
- **Post-training decision tuning** — orthogonal to model
  training.

## When NOT to use

- **Ranking metric matters more** — AUROC / AUPRC are threshold-
  independent.
- **Cost-sensitive decisions** — use expected utility with
  problem-specific cost matrix.
- **Multi-class** — extend to macro-/micro-averaged F-β or
  one-vs-rest tuning.

## Assumptions & caveats

- **Use a held-out set** — picking t on the training set is
  optimistic.
- **Prevalence-dependent** — the optimal t depends on the
  positive-class rate; re-tune on the target distribution.
- **Calibration** — badly miscalibrated probabilities need
  Platt / isotonic first for the sweep to be meaningful.
- **Continuity** — the sweep is discrete over the grid; refine
  around the max for precise t*.

## Related in this repo

- `youden-optimal-cutpoint` — J-statistic threshold.
- `roc-auc-analysis`, `calibration-scaling`,
  `discrimination-calibration` — classifier evaluation cousins.
- `decision-curve-analysis`, `cost-effectiveness-analysis` —
  utility-based decision tools.

## Run

```
python techniques/f1-optimal-threshold/python/f1_optimal_threshold.py
Rscript techniques/f1-optimal-threshold/r/f1_optimal_threshold.R
```

**Refs:** van Rijsbergen, C.J. *Information Retrieval*, Butterworths, 1979; Powers, D.M.W. "Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation." *J Machine Learning Technologies*, 2(1): 37-63, 2011.

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
