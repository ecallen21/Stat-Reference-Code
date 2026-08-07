# Tobit Regression (Reference §5.19)

Continuous outcome with a **pile-up at a known limit**: spending bottom-coded at 0, test scores top-coded at 100, sensor readings capped at instrument range. Latent-variable formulation:

```
y_i^* = X_i β + ε_i,     ε_i ~ N(0, σ²)
y_i   = max(L, min(y_i^*, U))       (observed censored value)
```

Standard Tobit (Tobin 1958): left-censoring at 0. Two-sided Tobit adds a right ceiling. Type-II ("Heckman selection" / heckit) uses separate selection and outcome equations.

## Log-likelihood

Three regimes:

```
y_i = L         Pr(y_i^* ≤ L) = Φ((L − X_i β) / σ)
y_i ∈ (L, U)    (1/σ) φ((y_i − X_i β) / σ)
y_i = U         Pr(y_i^* ≥ U) = 1 − Φ((U − X_i β) / σ)
```

MLE by BFGS on `(β, log σ)`.

## Why naive OLS fails

OLS on the observed `y` is **biased toward zero** (for right-censoring) or **inflated** (dropping censored obs) — the compressed tail changes the slope. Tobit un-does the censoring under the correctly-specified Normal error assumption.

## Files

- `python/tobit_regression.py` — from-scratch left / right / two-sided Tobit MLE. Demo (n = 400, 156 left-censored at 0): Tobit recovers β = (0.48, 1.12) close to truth (0.5, 1.0) and σ ≈ 0.99; naive OLS attenuates slope to (0.87, 0.70).
- `r/tobit_regression.R` — `AER::tobit` or `survival::survreg(Surv(y, y > 0, type = "left") ~ x, dist = "gaussian")`.

## When to use

- **Bottom-coded** expenditure, income, or utilization outcomes.
- **Top-coded** test scores, satisfaction ratings, sensor readings.
- Any continuous variable where the observed values pile up at a known boundary.

## When NOT to use

- **Two-part / hurdle**: if the zero-vs-positive process is truly a **different decision** from the intensity given participation, use a hurdle model (see `zero-inflated-regression`).
- **Non-Normal errors**: heavy-tailed outcomes need robust MLE or quantile approaches.
- **Multiple boundaries at unknown levels**: use interval-censored survival methods (see `interval-censored-survival`).

## Assumptions & caveats

- **Normal errors** — check with Q-Q plot on the uncensored residuals; the MLE is inconsistent under heteroscedasticity or heavy tails.
- **Marginal effects**: `∂E[y | X, L < y < U] / ∂x_k` involves both the coefficient and the Mill's ratio; report both raw `β` and the marginal effect.
- **Report the fraction censored** — Tobit's efficiency degrades sharply when > 80% of observations are at the boundary.

## Run

```
python techniques/tobit-regression/python/tobit_regression.py
Rscript techniques/tobit-regression/r/tobit_regression.R
```

**Refs:** Tobin, J. "Estimation of relationships for limited dependent variables." *Econometrica* 26(1), 24–36, 1958; Amemiya, T. "Tobit models: a survey." *J. Econometrics* 24(1–2), 3–61, 1984.

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
