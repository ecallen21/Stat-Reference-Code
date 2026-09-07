# Cross-Classified Random-Effects Model (CCREM) (Reference §8.19)

Raudenbush & Bryk (2002). When observations are grouped by **two or
more non-nested factors** (students within schools *and*
neighborhoods; patients within hospitals *and* primary-care
practices), put independent random effects on each grouping factor:

    y_ijk = μ + u_j + v_k + ε_ijk
    u_j ~ N(0, σ_u²),  v_k ~ N(0, σ_v²),  ε ~ N(0, σ_e²)

## Files

- `python/cross_classified_random_effects.py` — one-step EM for
  variance components from scratch, with both crossed factors +
  a mis-specified single-factor comparison. Demo (n=400, 12
  schools × 8 neighbourhoods): CCREM σ̂² ≈ (0.53, 0.21, 0.96) vs
  truth (0.64, 0.25, 1.00); mis-specified "schools only" absorbs
  neighborhood variance and inflates residual σ_e².
- `r/cross_classified_random_effects.R` — `lme4::lmer` with `(1 |
  school) + (1 | neighborhood)`, `glmmTMB`, `brms`, `MCMCglmm`
  (R); `statsmodels.MixedLM`, `pymer4`, `pymc / numpyro`
  (Python).

## When to use

- **Non-nested groupings** — students in schools + neighborhoods;
  patients across hospitals + PCPs.
- **Multiple membership** — a student attends more than one school
  (with weights).
- **Educational effectiveness / value-added modelling** —
  disentangle school vs neighborhood contributions.

## When NOT to use

- **Strictly nested designs** — a nested random-intercept model
  suffices (`(1 | school/classroom)`).
- **Very few levels per factor** — REML variance-component
  estimates are noisy with < 5 levels; use Bayesian priors.
- **You just care about population averages** — GEE with robust
  SE may suffice.

## Assumptions & caveats

- **Independence of random effects** — the classical form assumes
  `Cov(u_j, v_k) = 0`; extend with correlated random effects if
  needed.
- **Balanced-ish design** — extreme unbalance / small cells inflate
  computational cost and variance-component uncertainty.
- **Model comparison** — LRT for a random-effect requires
  the boundary correction (mixture of χ² at 0 and 1 df).
- **Level 1 residual autocorrelation** — not accounted for; use
  time-series or spatial extensions when relevant.

## Related in this repo

- `linear-mixed-models`, `generalized-linear-mixed-models` —
  parametric-mixed-model family.
- `bayesian-hierarchical-models` — the Bayesian analogue.
- `fay-herriot-small-area`, `poisson-gamma-empirical-bayes` —
  related small-area techniques.
- `multilevel-mediation`, `growth-curve-models` — extensions.

## Run

```
python techniques/cross-classified-random-effects/python/cross_classified_random_effects.py
Rscript techniques/cross-classified-random-effects/r/cross_classified_random_effects.R
```

**Refs:** Raudenbush, S.W. & Bryk, A.S. *Hierarchical Linear Models: Applications and Data Analysis Methods*, 2nd ed., Sage, 2002; Goldstein, H. *Multilevel Statistical Models*, 4th ed., Wiley, 2011.

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
