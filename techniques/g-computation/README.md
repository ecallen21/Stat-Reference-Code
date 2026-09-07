# G-Computation / Parametric G-Formula (Reference §15.20)

Robins (1986), Hernán & Robins (2020 ch 13). Standardise the
outcome model's predictions to a hypothetical population under
each treatment level:

```
ATE = E[ m(X, 1) − m(X, 0) ],   m(X, T) = E[Y | X, T]
```

**Outcome-model-based** counterpart of IPTW. For time-varying
treatments and confounders, extend to the **sequential g-formula**.

## When to use

- **Observational causal inference** where a reasonable outcome
  model is available.
- **Doubly-robust building block** — combined with IPTW gives
  AIPW; TMLE targets more efficiency.

## When NOT to use

- **Outcome-model misspecification risk** — g-computation alone
  is not robust; pair with IPTW → AIPW.
- **Extreme extrapolation** — predictions where no observations
  exist are unreliable.

## Files

- `python/g_computation.py` — linear outcome model + Hájek
  standardisation + bootstrap SE. Demo (n=2000, true ATE=0.5):
  point estimate **0.543 (95 % CI 0.454, 0.631)** — includes truth.
- `r/g_computation.R` — `stdReg::stdGlm`, `gfoRmula` (R);
  `zepid.causal.gformula.GFormula`, `causalinference`,
  custom (Python).

## Assumptions & caveats

- **Unconfoundedness** — the outcome model must capture all
  confounding.
- **Correct outcome model** — misspecification biases the ATE.
- **SEs via bootstrap** — plug-in inference under-estimates
  variance.
- **Positivity** — implicit; violations manifest as extrapolation.

## Related in this repo

- `iptw`, `aipw-doubly-robust`, `tmle-doubly-robust` — the causal-
  inference stack.
- `marginal-structural-model` — g-methods for time-varying
  confounding.

## Run

```
python techniques/g-computation/python/g_computation.py
Rscript techniques/g-computation/r/g_computation.R
```

**Refs:** Robins, J.M. "A new approach to causal inference in mortality studies with a sustained exposure period." *Mathematical Modelling*, 1986; Hernán, M.A. & Robins, J.M. *Causal Inference: What If*, Chapman & Hall/CRC, 2020 (ch 13).

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
