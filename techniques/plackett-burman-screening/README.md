# Plackett-Burman Screening Design (Reference §47.63)

Plackett & Burman (1946). Two-level fractional-factorial for
**screening** main effects of many factors with few runs. Design
matrix D is N × k, entries in {−1, +1}, with orthogonal columns
`D'D = N·I`. Main-effect estimates

    β̂_j = (1/N) Σᵢ D_{ij} · y_i.

Resolution III: main effects aliased with 2-factor interactions.
Ideal first-pass to prune large factor lists before running a
higher-resolution design.

## Files

- `python/plackett_burman_screening.py` — canonical N=12 PB
  design (cyclic first row from the 1946 paper) plus Hadamard
  fallback for N a power of 2. Demo (8 factors, N=12 runs, truly
  active at indices 1, 3, 6 with effects (3, −2, 1.5)):
  - orthogonality: D'D = 12 I ✔
  - estimated |effects| ranking's top-3 = {1, 3, 6} — the true
    active set.
- `r/plackett_burman_screening.R` — `FrF2::pb`,
  `DoE.base::oa.design` (R); `pyDOE`, from-scratch (Python).

## When to use

- **Screening 8–47 factors in ≤48 runs** — pilot studies before a
  full response-surface design.
- **Ruggedness testing** in analytical method validation.
- **Manufacturing process improvement** — identify vital few from
  trivial many.

## When NOT to use

- **Interactions matter** — resolution III conflates them with
  main effects; move to resolution IV+ design.
- **Continuous optimisation** — after screening, use CCD /
  Box-Behnken / D-optimal.
- **Nonlinear response** — 2-level PB misses curvature; add centre
  runs.

## Assumptions & caveats

- **Effect sparsity + effect hierarchy** — screening assumes few
  factors matter and main effects dominate.
- **Level scaling** — code low/high as ±1; results correspond to
  those coded units.
- **Randomisation** — always randomise run order to guard against
  time trends.
- **Analysis** — half-normal / normal probability plot of
  contrasts to spot active effects; complement with Lenth's method.

## Related in this repo

- `taguchi-methods`, `response-surface`, `latin-square-design`,
  `split-plot-design`, `balanced-incomplete-block-design` — DoE
  cousins.
- `multiple-testing-corrections`, `multiple-metrics-fdr`,
  `post-hoc-tests` — inference on many effects.
- `variable-selection`, `stability-selection`, `sure-independence-
  screening` — screening for observational data.
- `sensitivity-e-value`, `multi-vari-charts`, `pareto-charts`, `six-sigma-methods` —
  quality-improvement toolkit.

## Run

```
python techniques/plackett-burman-screening/python/plackett_burman_screening.py
Rscript techniques/plackett-burman-screening/r/plackett_burman_screening.R
```

**Refs:** Plackett, R.L. & Burman, J.P. "The design of optimum multifactorial experiments." *Biometrika* 33(4): 305-325, 1946; Box, G.E.P., Hunter, W.G. & Hunter, J.S. *Statistics for Experimenters*, 2nd ed., Wiley, 2005.

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
