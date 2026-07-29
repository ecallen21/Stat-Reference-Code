# Mixed-Effects Location-Scale Model (Reference §12.21)

Standard LMM assumes constant within-subject variance. **MELS** lets each subject have their own residual variance — captures individuals who are highly **volatile** vs. highly **consistent**, alongside their mean level.

## Model

```
y_{ij}   =  X_{ij}' β  +  u_{0i}  +  ε_{ij}
u_{0i}   ~  N(0, σ²_u)                                      (mean random effect)
ε_{ij}   ~  N(0, σ²_{ε,i})                                  (subject-specific residual)

log σ²_{ε,i}  =  γ_0  +  γ_1' Z_i  +  v_i                   (scale random effect)
v_i       ~  N(0, σ²_v)
```

So each subject gets **two random effects**: one for location (mean), one for scale (log-residual-variance). Useful when the *variability* of the outcome is itself a substantive outcome — mood variability, blood-pressure variability, day-to-day pain variability.

## Files

- `python/mixed_effects_location_scale.py` — **two-stage MELS** (LMM for mean → empirical per-subject residual variance → variance of log-SD across subjects). Recovers `σ_v = 0.48` vs. true 0.40 on the demo.
- `r/mixed_effects_location_scale.R` — pointer to `mixregls` (full joint MLE) and `nlme` with `varIdent` for a lighter alternative.

## Assumptions

- Same as LMM plus: residual variance follows a lognormal random-effect distribution across subjects.
- Enough within-subject observations to estimate each per-subject variance — rule of thumb `≥ 5` per subject.

## Run

```
python techniques/mixed-effects-location-scale/python/mixed_effects_location_scale.py
Rscript techniques/mixed-effects-location-scale/r/mixed_effects_location_scale.R
```

**Refs:** Hedeker, D., Mermelstein, R.J. & Demirtas, H. "An application of a mixed-effects location scale model for analysis of ecological momentary assessment (EMA) data." *Biometrics* 64(2), 627–634, 2008; Hedeker, D. & Nordgren, R. "MIXREGLS: A program for mixed-effects location scale analysis." *J. Stat. Soft.* 52(12), 1–38, 2013.

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
