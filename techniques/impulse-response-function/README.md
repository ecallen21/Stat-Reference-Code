# Impulse Response Function (Reference §47.307)

Sims (1980). For a VAR(p), trace the dynamic response of each
variable to a UNIT-SIZE structural shock over horizon
`h = 0, 1, …, H`:

```
y_t = Σ_{s≥0} Ψ_s · e_{t-s}          (MA representation)
IRF(h, j → k) = element (k, j) of Ψ_h · B_0⁻¹
```

## Files

- `python/impulse_response_function.py` — Companion-matrix
  form and IRF up to H=15. Demo VAR(1) with 0.5·e1 loading
  into y2: initial y1 ← e1 response 0.99; y2 ← e1 initial
  response 0.50; both decay by h=15 (stable VAR).
- `r/impulse_response_function.R` — `vars::irf`, `svars::irf`,
  `lpirfs` (R); statsmodels VAR.irf, from-scratch (Python).

## When to use

- **Macro / finance policy analysis** — response of GDP to
  monetary shock, etc.
- **Reporting SVAR results** — IRFs are the primary output.
- **Local projections (Jorda 2005)** — a direct-regression
  alternative.

## When NOT to use

- **Non-linear DGP** — VAR IRFs are linear; use generalised
  IRFs (Koop et al 1996) for asymmetric responses.
- **Time-varying dynamics** — TVP-VAR or TVP local projections.
- **Very short samples** — IRF confidence intervals become
  huge.

## Assumptions & caveats

- **Confidence bands** — bootstrap or delta-method; asymptotic
  intervals may under-cover.
- **Ordering-dependent** — Cholesky IRFs depend on the
  variable order; sign-restriction IRFs are ordering-free.
- **Cumulative IRFs** — often reported for long-run
  interpretation.
- **Impulse-response length** — 20-40 quarters typical for
  macro.

## Related in this repo

- `structural-var-svar` — the identification step.
- `variance-decomposition-fevd` — companion decomposition.
- `granger-causality`, `newey-west-hac` — related VAR tools.

## Run

```
python techniques/impulse-response-function/python/impulse_response_function.py
Rscript techniques/impulse-response-function/r/impulse_response_function.R
```

**Refs:** Sims, C.A. "Macroeconomics and reality." *Econometrica*, 48(1): 1-48, 1980; Jordà, Ò. "Estimation and inference of impulse responses by local projections." *AER*, 95(1): 161-182, 2005.

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
