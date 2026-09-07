# D-Optimal Design (Reference §17.19)

Fedorov (1972), Atkinson-Donev-Tobias (2007). Given a candidate set
of runs and a model parameterisation, choose `n` runs to **maximise
`det(X^T X)`** — equivalently, minimise the volume of the joint
confidence ellipsoid of the coefficient vector `β`.

## Algorithm

**Fedorov exchange** — greedy swap-in / swap-out heuristic:
starting from a random `n`-run subset, at each iteration swap the
run whose removal + candidate addition maximally increases
`det(X^T X)`.

**D-efficiency** relative to a benchmark:
`100 · (det/det_best)^(1/p) · (N_best/n)`.

## When to use

- **Constrained / irregular design regions** where classical
  factorial layouts don't fit.
- **Model already specified** — D-optimality assumes you know
  which terms matter.
- **Small run budget** — 9 D-optimal runs can capture most of the
  precision of 27 full-factorial runs.

## When NOT to use

- **Model mis-specification risk** — D-optimality can concentrate
  runs at design vertices, missing curvature. Consider I- or
  V-optimality.
- **Sequential learning** — Bayesian optimal designs update
  after each observation.

## Files

- `python/d_optimal_design.py` — Fedorov exchange (custom).
  Demo (3³ full = 27 candidates, main-effects + 2FI model with
  p=7): 9-run D-optimal subset achieves **158 % D-efficiency**
  when normalised per-run vs the 27-run full design — huge cost
  saving.
- `r/d_optimal_design.R` — `AlgDesign::optFederov`,
  `AlgDesign::optBlock`, `DoE.wrapper`, `skpr` (R); `pyDOE2`,
  `dexpy`, custom (Python).

## Assumptions & caveats

- **Candidate set** — must include all runs you're willing to
  perform; runs outside it never appear.
- **Model dependence** — a design optimal for one model may be
  poor for another; report the assumed model.
- **Local optima** — Fedorov is greedy; multi-start + Fedorov-
  Wynn refinements avoid them.
- **Bayesian D-optimality** — average `det(X^T Ω(β) X)` over the
  prior for nonlinear models.

## Related in this repo

- `fractional-factorial`, `response-surface`, `taguchi-methods`,
  `latin-hypercube-sampling` — companion DOE tools.

## Run

```
python techniques/d-optimal-design/python/d_optimal_design.py
Rscript techniques/d-optimal-design/r/d_optimal_design.R
```

**Refs:** Fedorov, V.V. *Theory of Optimal Experiments*, Academic Press, 1972; Atkinson, A.C., Donev, A.N., & Tobias, R.D. *Optimum Experimental Designs, with SAS*, Oxford University Press, 2007.

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
