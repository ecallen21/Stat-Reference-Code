# Multivariate Multiple Regression (Reference §9.20)

Regression with **multiple response variables** jointly:

```
Y (n × q) = X (n × p) · B (p × q) + E (n × q)
rows of E ~ N(0, Σ)                            residual covariance across responses
```

OLS estimator `B̂ = (XᵀX)⁻¹ Xᵀ Y` — the same as running `q` separate univariate OLS regressions, but MMR gives you a **joint residual covariance** for hypothesis testing across responses.

## Multivariate F-tests

Test `H_0: L B M = 0` for contrast matrices `L` (over predictors) and `M` (over responses). Four common test statistics:

- **Wilks' Λ** — likelihood-ratio flavor; classical default.
- **Pillai's trace** — most robust to non-normality and homoscedasticity violations.
- **Hotelling-Lawley trace** — closest to sum of individual F's.
- **Roy's largest root** — most powerful when only one direction matters.

All approximate F-distributions via Rao's formula.

## Files

- `python/multivariate_multiple_regression.py` — OLS MMR + Wilks Λ test with Rao's F approximation. Demo (n = 200, p = 3, q = 3, correlated errors): coefficients recovered to ~0.1 of truth; Wilks Λ = 0.092, F ≈ 149 (p ≈ 0) for the joint slopes-are-zero null.
- `r/multivariate_multiple_regression.R` — base `lm(Y ~ X)` with multivariate response + `car::Anova(fit, test.statistic = "Wilks")` for multivariate tests.

## When to use

- **Several correlated outcomes** measured on the same subjects — psychometric scales, biomarker panels, multi-response experiments.
- **Testing a joint effect** on all outcomes together, not just each in isolation.
- **Growth curves** / repeated measures cast as MMR with orthogonal polynomial time contrasts.
- **Reduced-rank regression** — related; enforces `B` to have low rank for parsimony.

## When to prefer separate univariate regressions

- Responses truly independent conditional on `X` — no efficiency gain from MMR.
- Only single-response hypothesis tests are of interest.

## Assumptions & caveats

- **Multivariate normal errors** — Pillai's trace is the most robust to deviations.
- **Same predictors for each response** — for different `X_j` per `Y_j`, use SUR (Seemingly Unrelated Regressions).
- **Report both** individual univariate coefficients (with their SEs) and the joint multivariate test.

## Related methods

- **SUR (Seemingly Unrelated Regressions)** — equation-specific predictors + joint GLS estimation.
- **Reduced-rank regression** — force `B` to have rank ≤ min(p, q).
- **Canonical correlation** — related; find linear combinations of Y and X with maximum correlation.

## Run

```
python techniques/multivariate-multiple-regression/python/multivariate_multiple_regression.py
Rscript techniques/multivariate-multiple-regression/r/multivariate_multiple_regression.R
```

**Refs:** Rencher, A.C. & Christensen, W.F. *Methods of Multivariate Analysis*, 3rd ed., Wiley, 2012 (Ch 10); Anderson, T.W. *An Introduction to Multivariate Statistical Analysis*, 3rd ed., Wiley, 2003.

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
