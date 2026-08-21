# Decision Curve Analysis (Reference §20.x extra)

Vickers & Elkin (2006). A **clinical utility** metric that combines
discrimination and calibration into a single quantity — the **net benefit** —
across a range of clinically plausible decision thresholds `p_t`:

```
NB(p_t) = TP / n − FP / n · [p_t / (1 − p_t)]
```

- `TP / n` — true-positive rate (benefit of correctly treating events).
- `FP / n` — false-positive rate (harm of treating non-events).
- `p_t / (1 − p_t)` — odds at the threshold, converting harm units into benefit units.

`p_t` reflects the clinician's / patient's trade-off: at `p_t = 0.15`, missing an event is ~5.7× worse than treating a non-event.

## Reference strategies

- **Treat all**: `NB_all = prevalence − (1 − prevalence) · p_t / (1 − p_t)`.
- **Treat none**: `NB = 0`.

A model is clinically useful iff its NB curve lies **above BOTH** references
across the relevant `p_t` range.

## Contrast with AUC / NRI / IDI

| Metric | Question |
|---|---|
| **AUC** | how well does the model rank events above non-events? |
| **Brier / calibration** | how accurate are the predicted probabilities? |
| **NRI / IDI** | does the new model reclassify better than the old? |
| **DCA** | does the model beat trivial "treat all" and "treat none" strategies at plausible thresholds? |

DCA is the only one that explicitly incorporates the **relative costs of
false positives and false negatives** — a model can have higher AUC yet lower
net benefit at the threshold the clinician actually cares about.

## When to use

- **Model translation** — deciding whether to deploy a risk-prediction model in the clinic.
- **New biomarker justification** — does adding it change clinical decisions net-positively?
- **Head-to-head model comparison** — often more informative than ΔAUC or NRI.
- **Guideline development** — pick thresholds that maximise NB across a target patient population.

## Files

- `python/decision_curve_analysis.py` — from-scratch NB computation + comparison across models and reference strategies. Demo (n=3000, prev 0.31, old model uses only x1, new model adds x2): new model dominates old and treat-all across the full plausible threshold range 5–60% (12 of 12 grid points).
- `r/decision_curve_analysis.R` — `rmda::decision_curve`, `dcurves::dca`, `DecisionCurve::dca`.

## Assumptions & caveats

- **Threshold range must be clinically defensible** — always tie `p_t` bounds to real clinical judgements. Presenting DCA over `[0, 1]` invites ambiguous conclusions.
- **Miscalibration hurts DCA** — a poorly-calibrated model that ranks well may still have negative NB. Calibrate first (`calibration-scaling`).
- **Prevalence dependency** — treat-all's NB shifts with the base rate; report DCA on the intended-use population, not a case-control subset.
- **Standard errors** — bootstrap for NB confidence bands (`rmda::decision_curve` with `bootstraps=500`).
- **Time-to-event outcomes** — need censoring-aware NB (Vickers 2008); use `dcurves` with `time` and `event` inputs.
- **Multiple decisions** — combine with net-benefit-integrated summaries (Vickers 2016) or apply DCA to each decision separately.

## Related in this repo

- `roc-auc-analysis` — discrimination.
- `calibration-scaling` — calibration slope, intercept, Brier decomposition.
- `nri-idi` — reclassification-based comparison.
- `harrell-c-index` — discrimination for censored outcomes.

## Run

```
python techniques/decision-curve-analysis/python/decision_curve_analysis.py
Rscript techniques/decision-curve-analysis/r/decision_curve_analysis.R
```

**Refs:** Vickers, A.J. & Elkin, E.B. "Decision curve analysis: A novel method for evaluating prediction models." *Med. Decis. Making* 26(6), 565–574, 2006; Vickers, A.J. & van Calster, B. "Net benefit approaches to the evaluation of prediction models, molecular markers, and diagnostic tests." *BMJ* 352, i6, 2016; Van Calster, B. et al. "Reporting and interpreting decision curve analysis: a guide for investigators." *Eur. Urol.* 74(6), 796–804, 2018.

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
