# Structural VAR (Reference §47.306)

Sims (1980); Blanchard & Quah (1989). A reduced-form VAR(p) is
transformed to a STRUCTURAL representation by identifying `B_0`
(or its inverse). Cholesky (recursive) identification is
simplest: order variables and take a lower-triangular
`B_0⁻¹` such that `B_0⁻¹ B_0⁻ᵀ = Σ`.

## Files

- `python/structural_var_svar.py` — Bivariate VAR(1) with a
  known structural DGP (5% cross-loading from e1 into y2).
  Cholesky recovers B_0⁻¹ that reproduces Σ (‖B_0⁻¹ B_0⁻ᵀ − Σ‖_F
  ≈ 0).
- `r/structural_var_svar.R` — `vars::SVAR`, `svars`,
  `bvartools` (R); `statsmodels.tsa.vector_ar.svar_model`,
  from-scratch (Python).

## When to use

- **Macroeconomic modelling** — identify monetary / fiscal /
  supply shocks.
- **Policy analysis** — trace effects of "unit" structural
  shocks.
- **Cross-country / cross-region VARs** with a well-defined
  ordering.

## When NOT to use

- **Ordering is contentious** — sign-restriction or
  narrative-identification SVAR is more robust.
- **Non-stationarity** — VECM handles cointegrated series.
- **Latent shocks** — dynamic factor models may be more
  appropriate.

## Assumptions & caveats

- **Cholesky ordering matters** — swap variables → different
  identification.
- **Stationarity** — VAR requires roots of A(z) inside the
  unit circle; test with ADF or Dickey-Fuller.
- **Lag length** — choose via AIC / BIC / HQ; too few lags →
  residual autocorrelation.
- **Alternative identifications** — long-run (Blanchard-Quah),
  short-run zero, sign restrictions, external instruments,
  or heteroskedasticity-based.

## Related in this repo

- `impulse-response-function`, `variance-decomposition-fevd`
  — downstream SVAR summaries.
- `var-cointegration`, `panel-cointegration` — cointegrated
  extensions.
- `granger-causality`, `stationarity-tests` — supporting
  diagnostics.

## Run

```
python techniques/structural-var-svar/python/structural_var_svar.py
Rscript techniques/structural-var-svar/r/structural_var_svar.R
```

**Refs:** Sims, C.A. "Macroeconomics and reality." *Econometrica*, 48(1): 1-48, 1980; Blanchard, O.J. and Quah, D. "The dynamic effects of aggregate demand and supply disturbances." *AER*, 79(4): 655-673, 1989.

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
