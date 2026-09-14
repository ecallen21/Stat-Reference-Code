# Sobol Variance-Based Sensitivity (Reference §47.334)

Sobol (1993); Saltelli et al (2008). Decompose the variance of
a scalar output `f(X_1, …, X_d)` into contributions from each
input and their interactions:

```
Var(Y) = Σ_i V_i + Σ_{i<j} V_{ij} + … + V_{1..d}
S_i    = V_i / Var(Y)                    first-order
S_Ti   = 1 − V_{~i} / Var(Y)              total-effect
```

`S_i` measures the main effect; `S_Ti − S_i` captures
interaction involvement.

## Files

- `python/sobol_variance_sensitivity.py` — Saltelli-style
  first-order + total-effect estimators on the Ishigami
  benchmark. Recovers S_i and S_Ti in the right ballpark of
  the analytic reference (S_i ≈ (0.31, 0.44, 0.00), S_Ti ≈
  (0.56, 0.44, 0.24)).
- `r/sobol_variance_sensitivity.R` — `sensitivity::sobol` /
  `sobolmartinez` (R); `SALib.analyze.sobol`, `UQpy`,
  from-scratch (Python).

## When to use

- **Uncertainty quantification** — engineering / climate /
  epi models.
- **Identifying important vs unimportant inputs** — factor
  fixing for follow-up experiments.
- **Detecting interaction** — S_Ti − S_i > 0 flags it.

## When NOT to use

- **Very expensive models** — N × (d+2) evaluations; use
  Morris for screening or PC expansion for surrogate.
- **Correlated inputs** — standard Sobol assumes
  independence; use Kucherenko / Owen for dependent inputs.
- **Discrete / categorical inputs** — need special handling.

## Assumptions & caveats

- **Independence of inputs** — standard formulas rely on it.
- **Sample size** — Saltelli's estimator needs N ~ 1000+ for
  precise indices.
- **Sobol / Halton sequences** improve variance vs uniform
  Monte Carlo.
- **Total-order estimator** (Jansen 1999) is more accurate
  than the older formula.

## Related in this repo

- `morris-elementary-effects` — cheap screening companion.
- `quasi-monte-carlo-sobol` — Sobol sequences (QMC).
- `sensitivity-e-value` — related sensitivity analysis for
  causal inference.

## Run

```
python techniques/sobol-variance-sensitivity/python/sobol_variance_sensitivity.py
Rscript techniques/sobol-variance-sensitivity/r/sobol_variance_sensitivity.R
```

**Refs:** Sobol, I.M. "Sensitivity analysis for non-linear mathematical models." *Math. Model. Comput. Exp.*, 1(4): 407-414, 1993; Saltelli, A. et al. *Global Sensitivity Analysis: The Primer*, Wiley, 2008.

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
