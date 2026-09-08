# Buckley-James Semi-Parametric AFT (Reference §11.32)

Buckley & James (1979). Semi-parametric AFT for right-censored
survival: fit `log T = X β + ε` without parameterising the residual
distribution.

## Iterative KM-imputation

1. Initialise `β` (e.g., OLS on complete cases).
2. Compute residuals `e_i = log T_i − X_iβ`.
3. Build KM of residuals; impute censored `e_i` with the
   conditional mean `E[e | e > e_censored]`.
4. Refit OLS on `y* = Xβ + e_imputed` → new `β`.
5. Iterate to convergence.

## Files

- `python/buckley_james_aft.py` — Buckley-James iteration from
  scratch with KM-based conditional mean of the residuals. Demo
  (n=400, 32% censoring, true β=(2.0, 0.5), Gumbel-min error):
  BJ estimate (1.76, 0.485) in 7 iterations — slope closely
  recovered, intercept slightly attenuated (typical BJ finite-
  sample bias).
- `r/buckley_james_aft.R` — `bujar` (boosting BJ), `rms::psm`
  parametric fallback (R); from-scratch numpy (Python).

## When to use

- **AFT interpretation** (change of survival time on log scale)
  **without a distributional assumption**.
- **Modest censoring** — up to ~50 %.
- **Alternative to Cox** when hazard non-proportional but AFT is
  plausible.

## When NOT to use

- **Heavy censoring** — residual KM tail poorly estimated;
  parametric AFT more stable.
- **Time-varying covariates** — BJ is time-invariant.
- **Very small n** — iterations can oscillate; use ranks (Ying
  1993) or bootstrap for SE.

## Assumptions & caveats

- **AFT residual iid** given `X` — required.
- **KM tail** must reach 0 for the conditional mean to be well
  defined; otherwise use restricted mean and bias correction.
- **Standard errors** — bootstrap; the naive OLS SEs on the
  imputed data are wrong.
- **Convergence** — usually 5-15 iterations; monitor `max|Δβ|`.

## Related in this repo

- `accelerated-failure-time`, `parametric-survival`,
  `cox-ph`, `cure-models` — survival cousins.
- `censored-quantile-regression` — quantile analogue.
- `piecewise-exponential-model` — Poisson-GLM cousin.

## Run

```
python techniques/buckley-james-aft/python/buckley_james_aft.py
Rscript techniques/buckley-james-aft/r/buckley_james_aft.R
```

**Refs:** Buckley, J. & James, I. "Linear regression with censored data." *Biometrika*, 66(3): 429-436, 1979; Ying, Z. "A large sample study of rank estimation for censored regression data." *Ann. Statist.*, 21(1): 76-99, 1993.

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
