# Heckman Selection Model (Reference §5.21)

**Selection bias**: `y` is observed only for a self-selected subset — labour-force participants, adopters, respondents to a survey, insured drivers filing claims. If selection is correlated with the outcome error, OLS on the observed subset is biased.

## Type-II Tobit (Heckman 1979)

```
Selection:  z_i^* = W_i γ + u_i,        d_i = I(z_i^* > 0)
Outcome:    y_i^* = X_i β  + ε_i,       observed only when d_i = 1
(u_i, ε_i)  ~ N(0, [[1, ρσ], [ρσ, σ²]])
```

`ρ ≠ 0` ⇒ selection bias; the observed subsample is not a random sample of the population.

## Two-step estimator

1. **Probit** regression of `d` on `W` → `γ̂`.
2. Compute the **inverse Mills ratio** `λ_i = φ(W_i γ̂) / Φ(W_i γ̂)`.
3. OLS regress `y` on `[X, λ]` for the selected subset:

```
E[y | X, d = 1] = X β + ρσ · λ_i
```

The coefficient on `λ` estimates `ρ · σ`; a t-test on it is a **test of selection bias**.

**Standard errors**: naive OLS SEs are wrong because `λ_i` is estimated. Heckman's closed-form correction adjusts them; alternatively, use joint MLE.

## Files

- `python/heckman_selection.py` — two-step Heckman with a probit first stage and OLS-with-Mills second stage. Demo (n = 800, ρ = 0.7): selection γ = (0.51, 1.02) matches truth (0.5, 1.0); inverse-Mills t = 8.36 correctly detects selection bias; naive OLS is biased in both intercept and slope.
- `r/heckman_selection.R` — `sampleSelection::heckit(...)` (canonical R implementation with correct SEs and MLE option).

## Exclusion restriction

Two-step Heckman is identified even when `X` and `W` are identical, but this relies on **nonlinearity of the inverse Mills ratio** — often unstable. Best practice: include at least one variable in `W` that is **not** in `X` (a selection-only covariate). This is the analog of the IV exclusion restriction.

## When to use

- **Wage regressions** — observed only for labour-force participants.
- **Insurance claims** — observed only for policyholders who filed.
- **Survey nonresponse** — outcomes only from respondents.
- Any observational study where the mechanism deciding "who shows up in the data" plausibly relates to the outcome.

## Related methods

- **Full-information MLE** — jointly estimates γ and β for efficiency and correct SEs.
- **Inverse-probability weighting** — reweight observed cases by 1 / P̂(d = 1 | W).
- **Multiple imputation** — treat unobserved y as missing.
- **Bounds analysis** (Manski) — non-parametric bounds without distributional assumptions.

## Assumptions & caveats

- **Bivariate normality** of the error pair is critical; violation biases both stages.
- **Exclusion restriction** — without it, results depend on the tail of Φ, which is fragile.
- **Report the Mills coefficient** always; its t-statistic is the actual test of selection bias — if not significant, revert to OLS.

## Run

```
python techniques/heckman-selection/python/heckman_selection.py
Rscript techniques/heckman-selection/r/heckman_selection.R
```

**Refs:** Heckman, J.J. "Sample selection bias as a specification error." *Econometrica* 47(1), 153–161, 1979; Puhani, P.A. "The Heckman correction for sample selection and its critique." *J. Econ. Surv.* 14(1), 53–68, 2000.

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
