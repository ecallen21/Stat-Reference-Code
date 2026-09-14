# FEVD — Forecast Error Variance Decomposition (Reference §47.308)

Sims (1980). Decompose the `h`-step forecast error variance of
each variable INTO CONTRIBUTIONS from each structural shock:

```
Var(y_{k, t+h} − ŷ) = Σ_j Σ_{s=0}^{h−1} (Ψ_s · B_0⁻¹)_{k, j}²
FEVD(k, j, h) = share attributable to shock j at horizon h
```

Complements IRFs: IRFs show direction and magnitude, FEVD
shows RELATIVE IMPORTANCE.

## Files

- `python/variance_decomposition_fevd.py` — Compute FEVD
  shares up to H=15 on the same bivariate VAR(1) used in the
  IRF demo. y1 is driven entirely by e1 (99.8%) since the
  DGP has no e2 contemporaneous term into y1; y2 receives
  both shocks (~54% / 46% at long horizons).
- `r/variance_decomposition_fevd.R` — `vars::fevd`,
  `svars::fevd` (R); statsmodels FEVD, from-scratch (Python).

## When to use

- **Reporting SVAR results** — pair IRFs with FEVD tables.
- **Assessing shock importance** — which shock drives
  business-cycle variation?
- **Model comparison** — compare FEVD across identification
  schemes.

## When NOT to use

- **Non-linear / non-Gaussian shocks** — FEVD assumes
  linearity + orthogonal shocks.
- **When you only care about direction** — IRFs suffice.
- **Very short horizons** — FEVD shares stabilise
  monotonically; interpret with the entire trajectory.

## Assumptions & caveats

- **Rows sum to 1** (100%) — each row is a partition of the
  forecast-error variance of one variable.
- **Ordering-dependent** — like IRFs, Cholesky FEVD depends
  on ordering.
- **Interpretation** — a shock with 5% FEVD at h=1 but 60%
  at h=20 means "long-run driver".
- **Confidence bands** — bootstrap for uncertainty.

## Related in this repo

- `structural-var-svar`, `impulse-response-function` —
  companion SVAR outputs.

## Run

```
python techniques/variance-decomposition-fevd/python/variance_decomposition_fevd.py
Rscript techniques/variance-decomposition-fevd/r/variance_decomposition_fevd.R
```

**Refs:** Sims, C.A. "Macroeconomics and reality." *Econometrica*, 48(1): 1-48, 1980; Lütkepohl, H. *New Introduction to Multiple Time Series Analysis*, Springer, 2005.

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
