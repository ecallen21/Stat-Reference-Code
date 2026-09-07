# Principal Stratification -- CACE / LATE (Reference §15.32)

Frangakis & Rubin (2002); Angrist, Imbens & Rubin (1996). When
treatment assignment `Z` is randomised but received treatment `D` is
not (imperfect compliance), the naive ITT dilutes the causal effect.
The Complier Average Causal Effect (CACE, a.k.a. LATE) recovers the
effect **among compliers** — units who would take treatment iff
assigned.

## Principal strata

| Stratum | `D(0)` | `D(1)` |
|---|---|---|
| Always-taker (A) | 1 | 1 |
| Complier (C)     | 0 | 1 |
| Defier (F)       | 1 | 0 |
| Never-taker (N)  | 0 | 0 |

Under **randomisation** + **exclusion restriction** + **monotonicity**
(no defiers):

    CACE = ITT / ITT_D  =  Cov(Y, Z) / Cov(D, Z)

Equivalent to 2SLS with `Z` as instrument for `D`.

## Files

- `python/principal_stratification_cace.py` — Wald estimator +
  two-stage least squares. Demo (n=8000, 40% compliers, τ=2.0):
  Estimated compliance shares C=0.41 / A=0.20 / N=0.39 recover truth;
  Wald CACE = +1.912 vs truth +2.00; ITT alone = +0.79 (diluted).
- `r/principal_stratification_cace.R` — `ivreg::ivreg`, `AER::ivreg`,
  `eStrata`, `noncomplyR` (R); `linearmodels.IV2SLS`,
  `statsmodels.sandbox.regression.gmm` (Python).

## When to use

- **Randomised trials with non-compliance** — encouragement designs,
  drop-outs, cross-overs.
- **IV with a randomised instrument** — e.g. lottery for
  vocational-training programmes.
- **Effectiveness vs efficacy** — ITT gives effectiveness, CACE gives
  the biological / behavioural effect for those who actually take it.

## When NOT to use

- **Two-sided non-compliance without randomisation** — you need the
  instrument to be exogenous.
- **Weak instrument** (ITT_D near 0) — CACE is undefined / explodes;
  report ITT + confidence set instead.
- **Interest in the population effect** — CACE speaks to compliers
  only; you cannot generalise it to always- or never-takers without
  strong extrapolation.

## Assumptions & caveats

- **Exclusion restriction** — `Z` affects `Y` only through `D`. Can
  fail if assignment itself has psychological / logistical effects.
- **Monotonicity** — no defiers. In some settings (e.g. behavioural
  ITT with backlash) this is dubious; **Manski bounds** give a
  partial-identification alternative.
- **Weak IV** — if ITT_D is small, Wald has huge variance and finite-sample
  bias toward the OLS estimate; use LIML or Anderson-Rubin CIs.
- **Covariate adjustment** — 2SLS with covariates preserves LATE
  interpretation only under conditional randomisation + monotonicity.

## Related in this repo

- `iv-2sls` — general IV framework.
- `iptw`, `aipw-doubly-robust` — for observational settings without an
  instrument.
- `manski-bounds` — sensitivity when monotonicity / exclusion fail.
- `mendelian-randomization` — genetic instruments for CACE-style effects.

## Run

```
python techniques/principal-stratification-cace/python/principal_stratification_cace.py
Rscript techniques/principal-stratification-cace/r/principal_stratification_cace.R
```

**Refs:** Frangakis, C.E. & Rubin, D.B. "Principal stratification in causal inference." *Biometrics*, 58(1): 21-29, 2002; Angrist, J.D., Imbens, G.W. & Rubin, D.B. "Identification of causal effects using instrumental variables." *JASA*, 91(434): 444-455, 1996.

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
