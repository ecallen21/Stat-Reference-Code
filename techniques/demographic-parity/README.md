# Demographic Parity / Statistical Parity (Reference Ch 31 Fairness)

**A classifier satisfies demographic parity when its selection rate is
equal across protected groups**, regardless of ground truth. Simplest and
most legally-entrenched fairness criterion.

## Formula

```
P( Ŷ = 1 | A = a )   equal for every group a.
```

Two headline summaries:

```
DP difference = max_a P(Ŷ = 1 | A = a) − min_a P(Ŷ = 1 | A = a)
DP ratio      = min_a P(Ŷ = 1 | A = a) / max_a P(Ŷ = 1 | A = a)
```

## Four-fifths rule (EEOC, 1978)

Under the **Uniform Guidelines on Employee Selection Procedures**, a
selection rate for any subgroup below `0.80 · max_selection_rate` is
presumptive evidence of *adverse impact*:

```
if DP ratio < 0.80  →  flagged.
```

Widely applied in US employment, credit, housing, and admissions
compliance audits.

## Why (and when) to use it

- **Legal / regulatory audit** — the four-fifths rule is baked into
  Title VII enforcement.
- **Aggregate representation goals** — mandatory quotas, admissions
  targets.
- **Very fast and label-free** — needs only predictions and `A`.

## When NOT to use it

- **Base rates truly differ** across groups — enforcing parity can
  reduce accuracy or set true positives against true negatives
  (Chouldechova 2017, Kleinberg 2016 impossibility).
- **Ground-truth-based fairness matters** — see `equal-opportunity`,
  `equalized-odds`, `calibration-parity`.
- **Individual fairness matters** — see `individual-fairness`,
  `counterfactual-fairness`.

## Files

- `python/demographic_parity.py` — from-scratch: `selection_rates`,
  `demographic_parity_diff`, `demographic_parity_ratio`,
  `four_fifths_pass`. Sweep decision thresholds on synthetic two-group
  data (base-rate imbalance) and show group-specific thresholds
  achieving DP ratio = 1 by construction.
- `r/demographic_parity.R` — `fairness` / `fairml` / `mlr3fairness`
  R packages; `fairlearn` (Python) analogues.

## Assumptions & caveats

- **Ignores Y** — a demographically parity-satisfying classifier can be
  arbitrarily inaccurate on some groups.
- **Selection-rate cutoff is arbitrary** — `0.80` is legal convention,
  not statistical.
- **Impossibility theorems** — with different base rates, DP,
  equalized-odds, and calibration cannot all hold (Chouldechova 2017).
- **Group definitions matter** — race / sex intersections change the
  DP ratio dramatically; report subgroup granularity you actually care
  about.

## Related in this repo

- `equalized-odds`, `equal-opportunity`, `calibration-parity` — error-
  rate–based alternatives.
- `disparate-impact` — the numerical formulation of the 4/5 rule.
- `reweighing-preprocessing`, `adversarial-debiasing`,
  `equalized-odds-postprocessing`, `exponentiated-gradient-reduction`
  — mitigations that aim at demographic parity or related criteria.
- `distributionally-robust-optimization` — worst-group risk (a
  different fairness formulation).

## Run

```
python techniques/demographic-parity/python/demographic_parity.py
Rscript techniques/demographic-parity/r/demographic_parity.R
```

**Refs:** Uniform Guidelines on Employee Selection Procedures, 29 CFR §1607 (1978); Feldman, M. et al. "Certifying and removing disparate impact." *KDD*, 2015; Chouldechova, A. "Fair prediction with disparate impact." *Big Data*, 2017; Barocas, S., Hardt, M. & Narayanan, A. *Fairness and Machine Learning*, 2019.

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
