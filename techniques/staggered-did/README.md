# Staggered Difference-in-Differences (Reference §35.19)

Classical two-way FE staggered DiD **fails** when treatment timing
varies AND treatment effects are heterogeneous (Goodman-Bacon 2021:
"forbidden comparisons" using already-treated units as controls for
newly-treated units).

Modern estimators (Callaway-Sant'Anna 2021, Sun-Abraham 2021,
de Chaisemartin-D'Haultfoeuille 2020) restrict controls to
**not-yet-treated** or **never-treated** units and aggregate up.

## Callaway-Sant'Anna ATT(g, t)

For treated cohort `g` and calendar time `t`:

```
ATT(g, t) = 𝔼[Y_t(g) − Y_t(0) | G = g]
```

Estimated as:

```
ATT_hat(g, t) = [Ȳ_t(g) − Ȳ_{g-1}(g)] − [Ȳ_t(control) − Ȳ_{g-1}(control)].
```

Then **aggregate** across cohorts / event times.

## Event-time aggregation

```
ATT(e) = Σ_g w_g · ATT(g, g + e)
```

with `w_g` proportional to cohort size or user-specified.

## When to use

- **Multiple treatment cohorts** with staggered adoption.
- **Heterogeneous treatment effects across cohorts / time**.
- **Event-study analyses** (pre-trend + post-treatment dynamics).

## When NOT to use

- **Single-cohort DiD** — classical `diff-in-diff` is fine.
- **No never-treated group AND treatment fully saturated** — controls
  vanish; consider synthetic DiD.

## Files

- `python/staggered_did.py` — from-scratch CS group-time ATTs +
  event-time aggregation using never-treated controls. Demo: 90
  units, cohorts at {never, t=3, t=5, t=7}, true per-period effect
  0.30. Event-time ATT estimates:
  - Pre (e = −3, −2): 0.06, −0.02 (near 0).
  - Post (e = 0..3): 0.32, 0.67, 0.90, 1.26 — recovers cumulative
    per-period effect.
- `r/staggered_did.R` — `did`, `fixest::sunab`, `bacondecomp`,
  `DIDmultiplegt` (R); `differences`, `pyfixest` (Python).

## Assumptions & caveats

- **Parallel trends** — pre-period ATTs should be ~ 0.
- **No anticipation** — units don't respond before treatment.
- **Never-treated vs not-yet-treated controls** — CS accommodates both;
  the not-yet-treated variant is used when never-treated is missing.
- **Aggregation weights** matter — simple average vs cohort-size
  weights vs group-time double-robust weights.
- **Standard errors** — cluster-bootstrap over unit; CS provide
  analytical + bootstrap SEs.

## Related in this repo

- `diff-in-diff` — classical 2×2 case.
- `synthetic-did`, `event-study` — sibling designs (this batch).
- `fixed-effects-panel` — the TWFE regressions that CS replaces.

## Run

```
python techniques/staggered-did/python/staggered_did.py
Rscript techniques/staggered-did/r/staggered_did.R
```

**Refs:** Callaway, B. & Sant'Anna, P.H.C. "Difference-in-differences with multiple time periods." *Journal of Econometrics*, 2021; Goodman-Bacon, A. "Difference-in-differences with variation in treatment timing." *Journal of Econometrics*, 2021; Sun, L. & Abraham, S. "Estimating dynamic treatment effects in event studies with heterogeneous treatment effects." *Journal of Econometrics*, 2021.

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
