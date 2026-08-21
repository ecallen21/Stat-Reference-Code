# Fractional Response / Fractional-Logit Regression (Reference §5.26)

Outcome in `[0, 1]` **including the boundary values 0 and 1** (unlike Beta regression, which requires strict interior). Common examples: fraction of budget spent, participation rate, percentage of votes.

## Papke-Wooldridge quasi-MLE (1996)

Model the conditional mean:

```
E[y | x] = G(xᵀ β)              G = logistic (or probit)
```

Maximize a Bernoulli-like log-likelihood **on the fractional outcome**:

```
ℓ(β) = Σ_i [ y_i log G(x_iᵀ β) + (1 − y_i) log(1 − G(x_iᵀ β)) ]
```

Even though `y` is not Bernoulli, this **quasi-MLE** is consistent for the conditional mean function. Standard errors need a **sandwich correction** (HC0 / HC1 / HC3) — see `sandwich-robust-se`.

## Fractional-logit vs Beta regression

|                     | Fractional-logit         | Beta regression                          |
|---------------------|--------------------------|------------------------------------------|
| Support             | `[0, 1]` including 0/1   | `(0, 1)` strictly interior               |
| Models              | conditional mean only    | full conditional distribution            |
| Estimator           | quasi-MLE                | full MLE                                 |
| Standard errors     | require sandwich         | model-based                              |
| Boundary handling   | native                   | need transform / inflated Beta           |

## Files

- `python/fractional_logit.py` — quasi-MLE + HC0 sandwich SEs. Demo (n = 500, 30 boundary zeros, 20 boundary ones): recovers all three coefficients close to truth; matches `statsmodels.GLM(family=Binomial, cov_type="HC0")` exactly.
- `r/fractional_logit.R` — `glm(y ~ x1 + x2, family = quasibinomial())` + `sandwich::vcovHC(type = "HC0")` for the canonical R workflow.

## When to use

- **Rates and proportions** in `[0, 1]` with boundary observations.
- **Firm-level participation rates** (e.g. 401(k) participation).
- **Budget shares**, allocation percentages.
- **Anything logistic-shaped** where you'd use a logistic regression but the outcome isn't 0/1.

## Contrast

- **Beta regression** (`beta-regression`) — model the full distribution, but boundary observations need special handling.
- **Zero/one-inflated Beta** — Beta mixture with point masses.
- **Tobit-like models** for censored continuous data (`tobit-regression`).
- **Ordinary logistic** — for 0/1 outcomes only.

## Assumptions & caveats

- **Correctly-specified mean function** — quasi-MLE consistency requires the logistic link is right.
- **Robust SEs mandatory** — model-based SEs are wrong under quasi-likelihood.
- **Interpretation**: coefficients are on the log-odds scale of the conditional mean, not the individual y.

## Run

```
python techniques/fractional-logit/python/fractional_logit.py
Rscript techniques/fractional-logit/r/fractional_logit.R
```

**Refs:** Papke, L.E. & Wooldridge, J.M. "Econometric methods for fractional response variables with an application to 401(k) plan participation rates." *J. Appl. Econom.* 11(6), 619–632, 1996; Ramalho, E.A., Ramalho, J.J.S. & Murteira, J.M.R. "Alternative estimating and testing empirical strategies for fractional regression models." *J. Econ. Surv.* 25(1), 19–68, 2011.

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
