# Taguchi Methods (Reference §17.15)

Taguchi (1986), Ross (1996). Robust-design methodology from
quality engineering:

- **Inner array** — control factors (things the engineer sets).
- **Outer array** — noise factors (things they can't).
- **Orthogonal arrays** (L4, L8, L9, L16, L18, L27) — small
  designs that keep main effects orthogonal.
- **Signal-to-noise ratios** (SNR) quantify robustness of the
  response:
  - Smaller-the-better: `−10 log(mean(y²))`
  - Larger-the-better: `−10 log(mean(1/y²))`
  - Nominal-the-best: `10 log(ȳ² / var(y))`

Choose control-factor levels that **maximise SNR** and tune the
mean separately.

## When to use

- **Robust-design experiments** — minimise sensitivity to
  uncontrolled noise while hitting a target mean.
- **Screening designs** in manufacturing / process engineering.
- **Small budget** — L9 needs only 9 runs for 4 factors × 3
  levels.

## When NOT to use

- **Interaction estimation** — Taguchi arrays sacrifice
  interactions to fit main effects in few runs.
- **Continuous factor optimisation** — use response-surface
  methodology instead.

## Files

- `python/taguchi_methods.py` — L4 / L9 array generators + three
  SNR variants + per-level SNR analysis. Demo (L9, 4 factors ×
  3 levels, 5 outer-array noise replicates each): recovered per-
  level mean SNR ranking suggests factor B level 1 and factor C
  level 1 maximise robustness.
- `r/taguchi_methods.R` — `DoE.base::oa.design`, `qualityTools`,
  `SixSigma` (R); `pyDOE2` + custom (Python).

## Assumptions & caveats

- **Confounded interactions** — L-array main effects are
  confounded with pairs of other factors' interactions; interpret
  cautiously.
- **SNR choice** must match the engineering objective.
- **Two-stage optimisation** — first maximise SNR (robustness),
  then adjust a "scaling" factor to hit the mean target.
- **Verify with confirmation run** — a chosen level combination
  should be re-run with sufficient replicates.

## Related in this repo

- `fractional-factorial`, `response-surface`, `d-optimal-design`
  — companion DOE tools.
- `six-sigma-methods`, `process-capability-indices` — quality
  cousins.

## Run

```
python techniques/taguchi-methods/python/taguchi_methods.py
Rscript techniques/taguchi-methods/r/taguchi_methods.R
```

**Refs:** Taguchi, G. *Introduction to Quality Engineering*, Asian Productivity Organization, 1986; Ross, P.J. *Taguchi Techniques for Quality Engineering*, 2nd ed., McGraw-Hill, 1996.

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
