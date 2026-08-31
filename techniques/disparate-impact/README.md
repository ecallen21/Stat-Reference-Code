# Disparate Impact (Reference Ch 31 Fairness)

The numerical formulation of the **US EEOC four-fifths rule**
(29 CFR §1607.4D, *Uniform Guidelines on Employee Selection
Procedures*, 1978). A widely-litigated legal / statistical audit
metric.

## Formula

```
DI  =  P( Ŷ = 1 | A = minority )  /  P( Ŷ = 1 | A = reference )
```

The **reference** is conventionally the group with the *maximum
selection rate*. Guidance:

- `DI ≥ 0.80` — presumptively compliant.
- `DI  < 0.80` — presumptive **adverse impact**; burden shifts to
  the employer to demonstrate job-relatedness / business necessity.

## Two-proportion confidence interval

Under the delta method on `log(DI)`:

```
SE(log DI) = √( (1 − p_a) / x_a  +  (1 − p_b) / x_b )
95% CI = exp( log DI ± 1.96 · SE(log DI) )
```

Reporting the CI alongside the point estimate is standard practice for
regulatory audits (Morris-Lobsenz 2001).

## Mitigation approaches

- **Threshold shift per group** — bring the minority selection rate up
  to reference (demo does this in a few lines).
- **Feldman disparate-impact remover** (2015) — geometrically
  interpolate feature distributions across groups.
- **Reweighing** (Kamiran-Calders 2012) — reweight training data.
- **Adversarial debiasing** (Zhang 2018) — in-training penalty.

## When to use

- **Regulatory compliance** for hiring, credit, housing, education
  (US Title VII, ECOA, Fair Housing Act).
- **First-pass fairness audit** — cheap, universally understood.

## When NOT to use

- **Ground-truth-based fairness matters** — see `equalized-odds`,
  `equal-opportunity`, `calibration-parity`.
- **DI ≥ 0.8 is not proof of no discrimination** — passing the rule
  can still mask individual-level unfairness.

## Files

- `python/disparate_impact.py` — from-scratch `disparate_impact_ratio`,
  `di_two_proportion_ci`. Synthetic two-group data with score-mean
  disparity: **DI(minority / majority) = 0.621 (CI [0.534, 0.721])
  → FAILS**; group-specific threshold shift lifts DI to 1.000 (CI
  [0.893, 1.120]) → PASSES.
- `r/disparate_impact.R` — `fairness` / `fairml` / `fairmodels`;
  `aif360` + `fairlearn` in Python.

## Assumptions & caveats

- **Sample size** — the four-fifths test is only presumptive; small
  samples get wide CIs, so a point estimate below 0.80 with CI
  crossing 1 is legally ambiguous.
- **Reference-group choice** — the EEOC uses the max-rate group as
  reference; sensitivity analyses report the DI against every other
  group.
- **Intersectional groups** (race × sex) can flip pass/fail flags —
  audit at the granularity that matters.
- **Post-hoc threshold shifts by group** are the crudest mitigation;
  legally sensitive in the US (CRA-1991 §106 prohibits some forms).
- **Selection-rate parity ≠ error-rate parity** — a DI-passing
  classifier can still have very different TPR/FPR by group.

## Related in this repo

- `demographic-parity` — the underlying statistical criterion.
- `equalized-odds`, `equal-opportunity`, `calibration-parity` —
  sibling criteria based on ground truth.
- `reweighing-preprocessing`, `adversarial-debiasing`,
  `equalized-odds-postprocessing`, `exponentiated-gradient-reduction`
  — mitigations that lift DI.
- `distributionally-robust-optimization` — worst-group risk lens.

## Run

```
python techniques/disparate-impact/python/disparate_impact.py
Rscript techniques/disparate-impact/r/disparate_impact.R
```

**Refs:** Uniform Guidelines on Employee Selection Procedures, 29 CFR §1607.4D (1978); Feldman, M. et al. "Certifying and removing disparate impact." *KDD*, 2015; Morris, S.B. & Lobsenz, R.E. "Significance testing of the four-fifths rule." *Personnel Psychology*, 2001.

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
