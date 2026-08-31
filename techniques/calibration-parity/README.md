# Calibration Parity / Predictive Parity (Reference Ch 31 Fairness)

The **ProPublica-COMPAS-debate criterion.** Chouldechova (2017)
formalised predictive parity, and Kleinberg (2017) proved that
calibration-by-group + equalized odds cannot both hold when base rates
differ.

## Two related definitions

**Predictive parity** (Chouldechova 2017):

```
P( Y = 1 | Ŷ = 1, A = a )    equal for every group a.
```

The positive-predictive-value (PPV) at the deployed threshold is the
same across groups.

**Calibration by group** (Kleinberg 2016 / Pleiss 2017):

```
P( Y = 1 | Ŷ = s, A = a )  =  s    for all scores s and every group a.
```

A stronger condition — the *score itself* is a well-calibrated
probability within each group.

## Kleinberg / Chouldechova impossibility

If group base rates differ and the classifier is not perfect:
**calibration by group + equalized odds cannot both hold**. Practitioners
must pick.

## When to use

- **Score-driven decisions** where users interpret `Ŷ` as an *actual
  probability* (medical risk scores, credit scores, actuarial models).
- **Regulated actuarial contexts** — insurance / lending where
  calibration is a legal expectation.

## When NOT to use

- **Equal error rates matter more** — pick `equalized-odds` /
  `equal-opportunity`.
- **Aggregate representation matters** — pick `demographic-parity`.
- **Base rates truly equal** — every fairness criterion coincides;
  pick whichever is easiest to audit.

## Files

- `python/calibration_parity.py` —
  1. Per-group PPV at fixed thresholds → predictive-parity difference.
  2. Per-group reliability diagram (5 bins) → per-group ECE.
  Well-calibrated per-group scores by construction (draw `s ~ Beta`,
  then `y | s ~ Bern(s)`) with different score distributions across
  groups: **per-group ECE ≈ 0.03 (both), yet predictive-parity
  difference 0.04 – 0.21** across thresholds — the Chouldechova /
  Kleinberg impossibility on display.
- `r/calibration_parity.R` — `fairness` / `fairml` /
  `rms::val.prob`; `fairlearn` + `aif360` in Python.

## Assumptions & caveats

- **PPV depends on the threshold** — Chouldechova's original argument
  fixes the threshold; audit at the operating point actually deployed.
- **Sample-size matters** — bin-wise reliability with small `n_a`
  is noisy; use isotonic-regression-based reliability curves or
  bootstrap CIs.
- **Not a substitute for calibration on the whole population** — a
  classifier can be group-calibrated yet globally miscalibrated after
  aggregating groups (Simpson-like reversal).
- **Post-hoc calibrators applied per group** (Pleiss 2017) can only
  achieve calibration parity by giving up on equalized odds.
- **Label bias** — if `Y` measurement differs across groups, both
  calibration parity and equalized odds are contaminated.

## Related in this repo

- `calibration-scaling` — the toolbox of Platt / isotonic / temperature
  calibrators used per group.
- `demographic-parity`, `equal-opportunity`, `equalized-odds` — the
  competing criteria.
- `equalized-odds-postprocessing` — the mitigation that intentionally
  gives up calibration parity to hit equalized odds.
- `reweighing-preprocessing`, `adversarial-debiasing` — mitigations
  aimed at demographic parity or related criteria.

## Run

```
python techniques/calibration-parity/python/calibration_parity.py
Rscript techniques/calibration-parity/r/calibration_parity.R
```

**Refs:** Chouldechova, A. "Fair prediction with disparate impact: a study of bias in recidivism prediction instruments." *Big Data*, 2017; Kleinberg, J., Mullainathan, S. & Raghavan, M. "Inherent trade-offs in the fair determination of risk scores." *ITCS*, 2017; Pleiss, G. et al. "On fairness and calibration." *NeurIPS*, 2017.

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
