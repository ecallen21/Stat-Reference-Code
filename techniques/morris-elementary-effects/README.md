# Morris Elementary Effects (Reference §47.335)

Morris (1991). Cheap screening sensitivity: for each factor
`i`, compute a finite difference (elementary effect) at random
starting points along a grid:

```
EE_i(x) = (f(x + Δ e_i) − f(x)) / Δ
μ_i*    = mean |EE_i|         (importance)
σ_i     = std EE_i            (interaction / nonlinearity)
```

Costs `r(d + 1)` evaluations, vs O(N (d + 2)) for Sobol.

## Files

- `python/morris_elementary_effects.py` — Radial-style Morris
  on toy `f(x) = 3 x1 + 5 x2² + 0.1 x1·x3 + 0 x4` with 20
  trajectories. Correctly ranks (linear x1, nonlinear x2,
  interaction-only x3, negligible x4).
- `r/morris_elementary_effects.R` — `sensitivity::morris` /
  `morrisMultOut` (R); `SALib.analyze.morris`,
  `UQpy.sensitivity.MorrisSensitivity`, from-scratch (Python).

## When to use

- **Screening step** before Sobol / EFAST — quickly identify
  the important inputs.
- **Expensive black-box models** — engineering simulators,
  climate models.
- **Nonlinear / interaction detection** — σ_i > 0 flags
  either nonlinearity or interaction.

## When NOT to use

- **When quantitative variance shares are needed** — use
  Sobol.
- **Categorical inputs** — need special handling.
- **Very small `d`** — a full factorial or Sobol may fit
  comfortably.

## Assumptions & caveats

- **p (levels)** — 4-8 typical; larger p → finer grid.
- **r (trajectories)** — 10-50 typical for screening.
- **μ vs μ***: μ (raw mean) can cancel out sign changes;
  Campolongo & Braddock recommend μ*.
- **Radial vs trajectory design** — trajectory design (used
  here) is Morris's original; radial (Campolongo 2011) is a
  variant.

## Related in this repo

- `sobol-variance-sensitivity` — quantitative follow-up.
- `sensitivity-e-value` — causal-inference sensitivity.
- `quasi-monte-carlo-sobol`, `latin-hypercube-sampling` —
  related sampling schemes.

## Run

```
python techniques/morris-elementary-effects/python/morris_elementary_effects.py
Rscript techniques/morris-elementary-effects/r/morris_elementary_effects.R
```

**Refs:** Morris, M.D. "Factorial sampling plans for preliminary computational experiments." *Technometrics*, 33(2): 161-174, 1991; Campolongo, F., Cariboni, J. and Saltelli, A. "An effective screening design for sensitivity analysis of large models." *Environ. Model. Softw.*, 22(10): 1509-1518, 2007.

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
