# Turnbull NPMLE for Interval-Censored Survival (Reference §11.30)

Turnbull (1976). Generalises Kaplan-Meier to observations known only
to lie in intervals `(L_i, R_i]` (exact, right-censored, or interval-
censored).

## Self-consistent EM

Iterate on masses `p_j` at candidate jump points (unique finite
`L`, `R` values):

    α_ij = 𝟙[(L_i, R_i] contains p_j] / Σ_k α_ik p_k
    p_j^{(t+1)} = (1/n) Σ_i α_ij · p_j^{(t)} / Σ_k α_ik p_k^{(t)}

Converges to the NPMLE of the survival function.

## Files

- `python/turnbull_interval_censored.py` — Turnbull EM from
  scratch. Demo (n=400, true T ~ Exp(rate 0.1), interval-censored
  via random check-ups): S(10) = 0.406 vs truth 0.368; S(20) =
  0.178 vs truth 0.135 — recovered close to true exponential
  survival despite coarse observation.
- `r/turnbull_interval_censored.R` — `icenReg::ic_np`,
  `interval::icfit`, `survival::survfit(interval2)` (R);
  `lifelines`, `scikit-survival` + custom (Python).

## When to use

- **Screening / periodic-follow-up** — HIV seroconversion, disease
  progression by imaging visits, patient diaries.
- **Grouped or coarsened event times** — questionnaire "did event
  happen since last visit?".
- **Mixed exact + interval + right-censored** data in one dataset.

## When NOT to use

- **All events are exact** — Kaplan-Meier is simpler.
- **Very sparse intervals** — Turnbull points may collapse; consider
  parametric interval-censored models (Weibull, log-normal).
- **Regression on covariates** — use `icenReg::ic_par / ic_sp` for
  covariate effects.

## Assumptions & caveats

- **Independent interval censoring** — mechanism generating (L, R)
  independent of T.
- **Jump-point granularity** — Turnbull mass placed only at
  observed L / R; smoothness needs kernel estimator afterward.
- **Standard errors** — the EM does not produce SEs directly;
  bootstrap or profile likelihood via `icenReg`.
- **Non-uniqueness** — the NPMLE mass is defined only up to
  intervals of "innermost" support; report Turnbull intervals.

## Related in this repo

- `kaplan-meier`, `nelson-aalen`, `interval-censored-survival` —
  survival cousins.
- `cure-models`, `frailty-models`, `parametric-survival` —
  parametric extensions.
- `case-crossover`, `piecewise-exponential-model` — related
  discrete-time / grouped survival.

## Run

```
python techniques/turnbull-interval-censored/python/turnbull_interval_censored.py
Rscript techniques/turnbull-interval-censored/r/turnbull_interval_censored.R
```

**Refs:** Turnbull, B.W. "The empirical distribution function with arbitrarily grouped, censored and truncated data." *JRSS-B*, 38(3): 290-295, 1976; Sun, J. *The Statistical Analysis of Interval-Censored Failure Time Data*, Springer, 2006.

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
