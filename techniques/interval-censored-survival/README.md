# Interval-Censored Survival (Reference §11.20)

Every subject's event time `T_i` is known only to lie in an interval `(L_i, R_i]`. Typical structure:

- `L_i` = last visit at which the subject was still event-free
- `R_i` = first visit at which the event was observed
- Left-censored: `L_i = 0`; right-censored: `R_i = ∞`

Common in **panel / periodic screening** data (screening intervals, HIV seroconversion visits, cognitive-decline assessments).

## Why not just use the midpoint?

Imputing `(L_i + R_i) / 2` or picking the left/right endpoint biases the survival curve — the bias grows with the width of the visit interval. Correct treatment either uses the Turnbull NPMLE or a parametric MLE with the interval likelihood.

## Turnbull NPMLE (Turnbull 1976)

Nonparametric MLE for `S(t)`. Find the disjoint **Turnbull intervals** where `S` may drop (from the intersection graph of the observed `(L_i, R_i]`), then EM / self-consistency iterate to convergence.

## Parametric interval-censored MLE

Under a chosen family (Weibull / log-normal / log-logistic),

```
L_i(θ) = S(L_i; θ) − S(R_i; θ)          product over subjects
```

Optimize the log-likelihood with any general-purpose optimizer.

## Regression

- `icenReg::ic_sp` (semi-parametric Cox-like), `icenReg::ic_par` (parametric AFT).
- `survival::survreg(Surv(L, R, type="interval2") ~ ...)` for the parametric AFT fit.

## Files

- `python/interval_censored_survival.py` — from-scratch Turnbull NPMLE via self-consistency EM plus Weibull MLE with the interval likelihood. On synthetic Weibull(1.6, 5) data with unit-width visit intervals, Weibull MLE recovers λ ≈ 4.9, k ≈ 1.7.
- `r/interval_censored_survival.R` — `icenReg::ic_np` and `icenReg::ic_par`; fallback to `survival::survfit(..., type="interval2")`.

## Assumptions

- Visit times are non-informative given covariates — subjects don't visit **because** they think an event happened.
- Independent censoring (subjects lost to follow-up drop out for reasons unrelated to the event process).

## Run

```
python techniques/interval-censored-survival/python/interval_censored_survival.py
Rscript techniques/interval-censored-survival/r/interval_censored_survival.R
```

**Refs:** Turnbull, B.W. "The empirical distribution function with arbitrarily grouped, censored, and truncated data." *J. R. Stat. Soc. B* 38(3), 290–295, 1976; Sun, J. *The Statistical Analysis of Interval-Censored Failure Time Data.* Springer, 2006.

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
