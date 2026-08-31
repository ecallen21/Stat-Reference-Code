# Equalized-Odds Post-Processing (Reference Ch 31 Fairness)

Hardt, Price & Srebro (2016) — **given any score-producing classifier**,
find a group-specific **randomised decision rule** that (a) matches both
TPR and FPR across groups, and (b) minimises accuracy loss relative to
the base classifier.

## Algorithm

```
1. Per-group ROC: sweep thresholds -> {(t, TPR_a, FPR_a)}.
2. Feasible set under randomisation = the CONVEX HULL of that ROC curve.
3. EO-feasible = intersection of the two group hulls.
4. Pick the (FPR*, TPR*) in the intersection that maximises overall
   accuracy under the population base rate.
5. For each group, decompose (FPR*, TPR*) as a randomised mixture of
   two thresholds:
       with prob p_a  use threshold t_a^low
       with prob 1-p_a use threshold t_a^high
```

The randomisation is essential — a single deterministic threshold per
group generally cannot hit the same (TPR, FPR) as another group's.

## When to use

- **You cannot retrain the base classifier** (proprietary model,
  regulatory freeze).
- **Equalized odds is the target** — for pure TPR match see
  `equal-opportunity`; for calibration-parity target see
  `calibration-parity` + Pleiss 2017.
- **Group labels available at deployment** — this method needs `A`
  at inference time.

## When NOT to use

- **`A` unavailable at inference** — no way to apply the per-group rule.
- **Randomised decisions are unacceptable** — legal / patient-safety
  regimes may forbid coin-flipping.
- **Small groups** — the empirical ROC hull is noisy; wider CI on
  achievable (TPR, FPR).

## Files

- `python/equalized_odds_postprocessing.py` — from-scratch: per-group
  ROC via a `n_thr = 50` threshold sweep + midpoint interpolation for
  the hull; feasible intersection (`tol = 0.02`); pick the accuracy-
  maximising EO-feasible point. Demo on two-group synthetic data:
  **base (0.85 / 0.12) and (0.74 / 0.33)** → **EO-feasible point
  (TPR = 0.70, FPR = 0.21)** with an overall-accuracy proxy of 0.756.
- `r/equalized_odds_postprocessing.R` — `fairness` R packages; `aif360`
  / `fairlearn.postprocessing.ThresholdOptimizer` in Python.

## Assumptions & caveats

- **Requires `A` at inference** — legally sensitive in some jurisdictions.
- **Randomisation** — often replaced by deterministic per-group
  thresholds in production, at the cost of exact EO.
- **Accuracy cost is real** — the closer the two base ROCs are to
  each other, the smaller the cost.
- **Sample noise** — bootstrap the ROC hull to get CIs on the
  achievable EO point.
- **Multi-class** — apply one-vs-rest; the theory generalises but
  the intersection is harder to describe.
- **Calibrated post-processor** (Pleiss 2017) trades EO for calibration
  parity; see `calibration-parity`.

## Related in this repo

- `equalized-odds`, `equal-opportunity` — the criteria this method
  enforces.
- `calibration-scaling`, `calibration-parity` — the alternative
  post-processing target.
- `reweighing-preprocessing`, `adversarial-debiasing`,
  `exponentiated-gradient-reduction` — pre- and in-training
  alternatives.

## Run

```
python techniques/equalized-odds-postprocessing/python/equalized_odds_postprocessing.py
Rscript techniques/equalized-odds-postprocessing/r/equalized_odds_postprocessing.R
```

**Refs:** Hardt, M., Price, E. & Srebro, N. "Equality of opportunity in supervised learning." *NeurIPS*, 2016; Pleiss, G. et al. "On fairness and calibration." *NeurIPS*, 2017.

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
