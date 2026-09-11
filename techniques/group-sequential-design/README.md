# Group Sequential Design (Reference §47.257)

Pocock (1977); O'Brien & Fleming (1979). Interim analyses at
`K` equally-spaced looks with STOPPING BOUNDARIES that preserve
overall Type-I error `α`:

- **Pocock**: `c_k = c_P` for all `k` — constant, easier to
  cross early
- **O'Brien-Fleming**: `c_k = c_OBF · √(K/k)` — very
  conservative early, ~ 1.96 at final look

## Files

- `python/group_sequential_design.py` — MC calibration of both
  boundary families using correlated Z-statistics. Demo:
  `α=0.05`, `K=4`. Pocock boundary `c=2.36` at every look;
  O'BF `c_OBF=2.02` giving per-look bounds
  `(4.05, 2.86, 2.34, 2.02)`. Under H₀ over 100 000 sims:
  Pocock Type-I=0.050 (mean look 3.91), O'BF Type-I=0.051
  (mean look 3.97, rarely stops early).
- `r/group_sequential_design.R` — `gsDesign::gsDesign`,
  `rpact::getDesignGroupSequential`, `ldbounds`, `clinfun`
  (R); rpact via reticulate, from-scratch (Python).

## When to use

- **Confirmatory trials** with pre-planned interim looks for
  efficacy or safety.
- **Time-to-event endpoints** where events accumulate slowly
  — interim looks let you stop early for benefit.
- **Rare-disease trials** where one wants to minimise expected
  N under H₁ (O'BF conserves N; Pocock stops earlier).

## When NOT to use

- **Data-driven look schedules** — use Lan-DeMets alpha
  spending instead.
- **Adaptive re-estimation of sample size** — combine with
  CHW-weighted Z (see sample-size-reestimation).
- **Exploratory / phase-II** where a fixed-sample design is
  simpler.

## Assumptions & caveats

- **Independent increments** of the test statistic (holds for
  Z / logrank / most Wald-type stats).
- **Fixed information fraction** — Pocock/OBF are strict; use
  Lan-DeMets α-spending if look times drift.
- **P-value bias** — reported p at an interim stop needs
  adjustment (stagewise-ordered p-value).
- **Bias in effect estimate** — over-estimates the effect when
  stopping early for benefit (MUE / conditional-bias
  corrections).

## Related in this repo

- `alpha-spending-lan-demets` — the flexible-schedule cousin.
- `conditional-power-futility` — futility rules within a GSD.
- `sample-size-reestimation` — adaptive N combined with GSD.
- `sequential-analysis` — SPRT / SPRT-flavoured methods.

## Run

```
python techniques/group-sequential-design/python/group_sequential_design.py
Rscript techniques/group-sequential-design/r/group_sequential_design.R
```

**Refs:** Pocock, S.J. "Group sequential methods in the design and analysis of clinical trials." *Biometrika*, 64(2): 191-199, 1977; O'Brien, P.C. and Fleming, T.R. "A multiple testing procedure for clinical trials." *Biometrics*, 35(3): 549-556, 1979.

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
