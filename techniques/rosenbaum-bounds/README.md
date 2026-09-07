# Rosenbaum Bounds (Reference §15.23)

Rosenbaum (2002, 2010). Sensitivity analysis for **unmeasured
confounding** in matched-pair observational studies. Report the
smallest **`Γ`** (odds ratio of unmeasured-confounder → treatment
within matched pairs) at which the study's conclusion becomes
non-significant.

## Formulation

For binary matched-pair outcomes with `n_+` discordant pairs
favouring treatment and `n_−` favouring control:

- Under `Γ = 1` (no hidden bias), pair discordance is symmetric
  and the McNemar test applies.
- Under `Γ > 1`, the probability of a treatment-favouring pair can
  be as high as `Γ / (1 + Γ)`; upper-bound p-value uses this
  Bernoulli null.

## When to use

- **Matched observational study** where randomisation is not
  possible.
- **Sensitivity analysis** paired with the E-value or Cornfield
  bounds for triangulation.

## When NOT to use

- **Unmatched cohorts** — different sensitivity frameworks
  (E-value, `Γ` for full-cohort estimators).
- **Continuous outcomes** — extend via `rbounds::hlsens` or
  `sensitivitymv` (Rosenbaum multi-variate versions).

## Files

- `python/rosenbaum_bounds.py` — upper-bound p-value over a `Γ`
  grid + critical `Γ` at which p first exceeds α. Demo (60
  discordant pairs, 45 favour treatment): p_upper 1e-4 at Γ=1,
  0.05 crossed at **Γ ≈ 1.8** — an unmeasured confounder ~1.8×
  more prevalent in one arm would overturn the conclusion.
- `r/rosenbaum_bounds.R` — `rbounds::binarysens` / `hlsens`,
  `sensitivitymv`, `sensitivitymw`, `EValue` (R); custom + scipy
  (Python).

## Assumptions & caveats

- **Matched pairs** — the sensitivity refers to within-pair
  imbalance from unmeasured factors.
- **Symmetric perturbation** — assumed direction of `Γ` matters
  for one-sided tests; report both.
- **E-value** (VanderWeele-Ding 2017) is a cousin for unmatched
  cohort studies — see `sensitivity-e-value`.
- **`Γ` is not directly clinical** — interpret via known measured
  covariates ("Γ = 2 is like doubling smoking status
  contribution").

## Related in this repo

- `sensitivity-e-value` — E-value alternative.
- `propensity-score-matching` — matched-pair construction.
- `mahalanobis-distance` / `genetic-matching` (if present) —
  matching diagnostics.

## Run

```
python techniques/rosenbaum-bounds/python/rosenbaum_bounds.py
Rscript techniques/rosenbaum-bounds/r/rosenbaum_bounds.R
```

**Refs:** Rosenbaum, P.R. *Observational Studies*, 2nd ed., Springer, 2002; Rosenbaum, P.R. *Design of Observational Studies*, Springer, 2010.

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
