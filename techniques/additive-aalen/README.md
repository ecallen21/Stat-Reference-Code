# Aalen Additive-Hazards Regression (Reference §11.14)

Alternative to Cox's multiplicative hazards model. The Cox model assumes covariate effects multiply the baseline hazard proportionally; Aalen's additive model lets them **add on** and vary with time:

```
Cox   :  h_i(t) = h_0(t) exp(X_i β)              multiplicative, constant β
Aalen :  h_i(t) = β_0(t) + X_iᵀ β(t)              additive, time-varying β(t)
```

Each coefficient `β_k(t)` is a nonparametric function of time, estimated by least-squares increments at each observed event time (Aalen 1980).

## Estimation

```
dB(t) = (X_Rᵀ X_R)⁻¹ X_Rᵀ dN(t)                (X_R = design on the risk set)
B(t) = ∫_0^t β(u) du                            (cumulative regression function)
```

Report the **cumulative** `B_k(t)` as a step function. Slope over any interval = average `β_k(t)` in that interval. Testing constant-zero effect: sup-norm or integrated-`B` statistics.

## Files

- `python/additive_aalen.py` — least-squares increments of `B(t)` at each event time; sup-norm z-test per covariate. Demo (n = 300, true `β_1 = 0.05`): estimated average slope of `B_1(t)` over full follow-up = 0.056; sup-norm z = 2.43 rejects the constant-zero null.
- `r/additive_aalen.R` — `timereg::aalen` (Scheike-Martinussen; supports both constant and time-varying effects in one call) or `survival::aareg`.

## When to use

- **Time-varying effects** — covariate influence changes with follow-up (age-adjusted vaccine efficacy that wanes).
- **Additive-scale interpretation** — "how many extra deaths per person-year does this covariate cause?" is directly on the hazard scale.
- **Comparison with Cox** — a Cox model with proportional-hazards violation may still fit fine as Aalen.

## Testing constant vs time-varying effects

`timereg::aalen` fits a **semi-parametric** version: some covariates constant, others time-varying. Compare with a Kolmogorov-Smirnov-style test on the residuals to decide which covariates need time-varying `β_k(t)`.

## Assumptions & caveats

- **Hazards can go negative** — the additive scale doesn't constrain `h(t) ≥ 0`. In finite samples with large negative `β_k`, fitted hazards can be negative; check.
- **Design-matrix conditioning** — small risk sets late in follow-up make `(X_Rᵀ X_R)⁻¹` unstable; truncate the last few event times if needed.
- **Interpretation**: `β_k(t)` has units of `[hazard per unit change in X_k]` — units matter, standardize continuous covariates.

## Run

```
python techniques/additive-aalen/python/additive_aalen.py
Rscript techniques/additive-aalen/r/additive_aalen.R
```

**Refs:** Aalen, O.O. "A model for nonparametric regression analysis of counting processes." In *Mathematical Statistics and Probability Theory*, Springer, 1980; Scheike, T.H. & Zhang, M.-J. "An additive-multiplicative Cox-Aalen regression model." *Scand. J. Stat.* 30(3), 493–508, 2003; Martinussen, T. & Scheike, T.H. *Dynamic Regression Models for Survival Data*, Springer, 2006.

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
