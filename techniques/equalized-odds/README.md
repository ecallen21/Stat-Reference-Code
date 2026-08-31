# Equalized Odds (Reference Ch 31 Fairness)

**Equal true- AND false-positive rates across groups.** Hardt, Price &
Srebro (2016) proposed equalized odds as the "conditional-on-Y" fairness
criterion — a classifier can differ in selection *rate* across groups
(if base rates differ), but must not have systematically different error
rates within either label.

## Definition

```
P( Ŷ = 1 | Y = y, A = a )    equal for every group a, both y ∈ {0, 1}.
```

Two headline summaries:

```
EO difference = max( |TPR_a − TPR_b|,  |FPR_a − FPR_b| )
EO ratio      = min per-group  /  max per-group     (of TPR and FPR)
```

## Contrast with sibling criteria

| Criterion               | Matches                                        |
|-------------------------|------------------------------------------------|
| Demographic parity      | `P(Ŷ = 1 | A)`                                 |
| Equal opportunity       | `TPR` only                                     |
| Equalized odds          | `TPR` and `FPR`                                |
| Calibration parity      | `P(Y = 1 | Ŷ = s, A) ≡ P(Y = 1 | Ŷ = s)`      |

## Impossibility (Chouldechova 2017, Kleinberg 2016)

With unequal group base rates, equalized odds + calibration parity
cannot both hold. Pick the criterion that matches your harm model.

## When to use

- **Error rates matter across groups** — medical screening, credit
  underwriting, risk-assessment tools.
- **Anti-discrimination law** in some domains references *equal error
  rates* rather than equal selection rates.
- **Auditing an existing classifier** — a single reportable summary.

## Files

- `python/equalized_odds.py` — from-scratch `per_group_rates`,
  `equalized_odds_diff`, `equalized_odds_ratio`. Demo on synthetic
  two-group data (base rates 0.48 vs 0.21; noisier scores for the
  minority): threshold sweep shows the EO gap is fairly *stable*
  across thresholds (0.12 – 0.14) because the underlying score-quality
  asymmetry is baked into the classifier.
- `r/equalized_odds.R` — `fairness` / `fairml` / `mlr3fairness`;
  `fairlearn.metrics.equalized_odds_*` in Python.

## Assumptions & caveats

- **Ground truth `Y` required** — not applicable to unsupervised or
  loosely-labelled prediction.
- **A single threshold rarely equalises both TPR *and* FPR** — see
  `equalized-odds-postprocessing` for group-specific-threshold /
  randomised-decision recipes (Hardt 2016).
- **Doesn't imply calibration parity** — the two conflict under
  base-rate imbalance.
- **Difference vs ratio** — the difference is easier to interpret;
  the ratio is more sensitive when TPRs are small.
- **Multi-class extension** — apply per-class one-vs-rest.

## Related in this repo

- `demographic-parity`, `equal-opportunity`, `calibration-parity` —
  sibling fairness criteria.
- `equalized-odds-postprocessing` — Hardt's mitigation to enforce EO.
- `adversarial-debiasing`, `exponentiated-gradient-reduction` — in-
  training mitigations.
- `distributionally-robust-optimization` — worst-group risk (a
  different fairness lens).

## Run

```
python techniques/equalized-odds/python/equalized_odds.py
Rscript techniques/equalized-odds/r/equalized_odds.R
```

**Refs:** Hardt, M., Price, E. & Srebro, N. "Equality of opportunity in supervised learning." *NeurIPS*, 2016; Chouldechova, A. "Fair prediction with disparate impact." *Big Data*, 2017; Kleinberg, J., Mullainathan, S. & Raghavan, M. "Inherent trade-offs in the fair determination of risk scores." *ITCS*, 2017.

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
