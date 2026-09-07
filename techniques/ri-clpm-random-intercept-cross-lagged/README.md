# Random-Intercept Cross-Lagged Panel Model (Reference §20.30)

Hamaker, Kuiper & Grasman (2015). Separates **between-person**
(stable trait) variance from **within-person** dynamics — the
classical CLPM conflates them.

## Model

For two variables `X`, `Y` measured at `t = 1, …, T`:

    x_t = α_x + u_i + x*_t
    y_t = α_y + v_i + y*_t
    x*_t = φ_xx · x*_{t−1} + φ_xy · y*_{t−1} + e_x
    y*_t = φ_yx · x*_{t−1} + φ_yy · y*_{t−1} + e_y

`(u_i, v_i)` are stable trait intercepts; `(x*_t, y*_t)` are
within-person deviations. **`φ_xy` is the within-person effect** of
`Y` at `t − 1` on `X` at `t`, after the trait is removed.

## Files

- `python/ri_clpm_random_intercept_cross_lagged.py` — classical
  CLPM vs subject-centred RI-CLPM VAR from scratch. Demo (n=400,
  T=6, true φ_xy=0.15, φ_yx=0): classical CLPM says φ_xx=0.98,
  φ_xy=0 (trait variance dominates); RI-CLPM recovers φ_xy=+0.139
  and φ_yx≈0 (correct).
- `r/ri_clpm_random_intercept_cross_lagged.R` — `lavaan::sem`,
  `OpenMx`, `brms`, `psychonetrics` (R); `semopy`, `pymc`,
  `statsmodels.MixedLM` (Python).

## When to use

- **Panel data** — repeated measures on the same units.
- **Causal-ish claims about within-person dynamics** — teachers,
  students, patients, employees.
- **Ambulatory / daily-diary studies** — many time points; RI-CLPM
  correctly disentangles trait vs state.

## When NOT to use

- **Cross-sectional data** — no within-person variation.
- **Very short panels** (T=2) — RI-CLPM under-identified; classical
  CLPM is the ceiling.
- **You only care about group-level trends** — a mixed model on the
  outcome suffices.

## Assumptions & caveats

- **Stationarity of within-person process** — classical RI-CLPM
  assumes it; extensions (Mulder-Hamaker 2020) relax.
- **Measurement invariance across time** — same construct measured
  the same way at each wave.
- **Missing waves** — FIML in lavaan / OpenMx handles MCAR/MAR.
- **Naïve centring caveat** — centring by subject removes the trait
  but eats a df; SEM (lavaan) does it implicitly and gives correct
  SEs.

## Related in this repo

- `cross-lagged-panel`, `path-analysis`, `cfa-confirmatory-factor`
  — SEM cousins.
- `linear-mixed-models`, `generalized-linear-mixed-models`,
  `mixed-effects-location-scale` — mixed-model alternatives.
- `dynamic-time-warping`, `state-space-kalman` — other time-panel
  methods.

## Run

```
python techniques/ri-clpm-random-intercept-cross-lagged/python/ri_clpm_random_intercept_cross_lagged.py
Rscript techniques/ri-clpm-random-intercept-cross-lagged/r/ri_clpm_random_intercept_cross_lagged.R
```

**Refs:** Hamaker, E.L., Kuiper, R.M. & Grasman, R.P.P.P. "A critique of the cross-lagged panel model." *Psychological Methods*, 20(1): 102-116, 2015; Mulder, J.D. & Hamaker, E.L. "Three extensions of the random intercept cross-lagged panel model." *Structural Equation Modeling*, 28(4): 638-648, 2020.

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
