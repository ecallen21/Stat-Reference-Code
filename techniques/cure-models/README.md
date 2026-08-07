# Mixture Cure Models (Reference §11.22)

Some survival populations contain a fraction `π` of subjects who will **never** experience the event — cancer patients cured by successful treatment, subjects who never develop a rare condition. Standard survival models force `S(∞) = 0`, biasing estimates when a plateau appears in the Kaplan-Meier curve.

## Berkson-Gage mixture (Farewell 1982)

```
S_pop(t) = π + (1 − π) · S_u(t)
π         : cure probability
S_u(t)    : survival of the uncured subgroup (proper distribution → 0)
```

## Regression version

```
logit π(x) = α_0 + αᵀ x          (incidence submodel — who gets cured?)
S_u(t | z) = Weibull / Cox / ... (latency submodel — timing among uncured)
```

Two clinically meaningful effects: covariates on `π` predict long-term cure; covariates on `S_u(t)` predict timing.

## Estimation

- **EM algorithm** — E-step computes cure-status posteriors `w_i = Pr(uncured | y_i, δ_i)`; M-step is a weighted logistic + weighted survival fit.
- **Direct MLE** — BFGS on the mixture likelihood. Simple for small examples.
- **Full Bayesian** — `flexsurvcure::flexsurvcure` with `bayes = TRUE`; posterior separates cure vs latency uncertainty cleanly.

## Files

- `python/cure_models.py` — Weibull mixture cure via direct BFGS MLE, with optional covariates on the cure probability. Demo (n = 500, true π = 0.35, Weibull(1.4, 3)): recovers `π̂ = 0.29`, shape 1.37, scale 2.92; covariate version recovers `α = (−0.47, 0.77)` (true (−0.5, 0.7)).
- `r/cure_models.R` — `flexsurvcure::flexsurvcure` or `smcure::smcure`.

## When to use

- **KM curve plateaus well above 0** — visual evidence of a cured subgroup.
- **Cancer registries** with long follow-up: cure fractions are the outcome of interest.
- **Prevention trials** where a substantial fraction are true non-responders.
- **Any application** where "how many will be cured?" and "how long for the rest?" are separately meaningful.

## When NOT to use

- **Short follow-up** — cure fraction not identifiable without observing the plateau.
- **No plateau in KM** — the model still fits but reduces to the ordinary survival model; report the standard fit instead.
- **Two-population model unclear** — biology / substantive story should support the mixture interpretation, not just numerics.

## Assumptions & caveats

- **Identifiability** — needs enough censored subjects at long times to distinguish cured from very-long-latency uncured. Follow-up ≥ 3× the median event time is a rule of thumb.
- **Latency distribution matters** — a wrong parametric family biases both `π̂` and `β`; consider `smcure::smcure(model = "ph")` for a Cox-based latency (semiparametric).
- **Report both submodels** — cure regression coefficients and latency parameters have distinct interpretations.

## Run

```
python techniques/cure-models/python/cure_models.py
Rscript techniques/cure-models/r/cure_models.R
```

**Refs:** Berkson, J. & Gage, R.P. "Survival curve for cancer patients following treatment." *JASA* 47(259), 501–515, 1952; Farewell, V.T. "The use of mixture models for the analysis of survival data with long-term survivors." *Biometrics* 38(4), 1041–1046, 1982; Peng, Y. & Yu, B. *Cure Models: Methods, Applications, and Implementation*, CRC, 2021.

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
