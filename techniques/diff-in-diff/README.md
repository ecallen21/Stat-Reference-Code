# Difference-in-Differences (Reference §15.4)

Estimates a treatment effect from panel or repeated cross-section data by comparing pre-post changes in treated vs control groups. Under **parallel trends**, DID identifies the **average treatment effect on the treated (ATT)**:

```
DID = (ȳ_{treated, post} − ȳ_{treated, pre})
    − (ȳ_{control, post} − ȳ_{control, pre})
```

## Canonical 2×2 regression

```
y_it = α + β · treated_i + γ · post_t + τ · (treated_i · post_t) + ε_it
```

`τ` is the ATT estimate; `β` captures baseline differences, `γ` captures common time trends.

## Two-way fixed-effects (TWFE) generalization

```
y_it = α_i + γ_t + τ · D_it + ε_it
```

Absorbs unit-specific baselines and period shocks. `D_it = 1` when unit `i` is treated at time `t`, allowing staggered adoption.

## Modern warnings (heterogeneous / staggered)

With heterogeneous treatment effects and staggered adoption, **TWFE gives a weighted average of ATTs with sometimes-negative weights** (Goodman-Bacon 2021; de Chaisemartin & D'Haultfœuille 2020). Modern alternatives:

- **Callaway-Sant'Anna (2021)** — event-time ATT(g, t) plus aggregations.
- **Sun-Abraham (2021)** — interaction-weighted event study.
- **Borusyak-Jaravel-Spiess (2024)** — imputation-based DID.

Report both classical TWFE and one of these when adoption is staggered.

## Event-study specification

```
y_it = α_i + γ_t + Σ_k τ_k · D_{i, t − event_i = k} + ε_it
```

Pre-treatment `τ_{k<0}` should be near zero — a **placebo test** for parallel trends.

## Files

- `python/diff_in_diff.py` — 2×2 DID via interaction regression + TWFE-DID via iterated within-transform. Demo: 2×2 recovers ATT = 1.73 (true 1.5, within 2 SE); TWFE on a 40-unit / 6-period staggered panel recovers τ̂ = 2.11 (true 2.0).
- `r/diff_in_diff.R` — `lm(y ~ treated * post)` and `fixest::feols(y ~ D | unit + time)`; for staggered adoption use `did::att_gt(...)`.

## When to use

- **Policy evaluation** with a natural control group and pre-treatment baseline.
- **Quasi-experimental** studies where randomization isn't feasible but treatment timing varies.
- **Marketing / A-B rollouts** phased across markets.

## Key assumption: parallel trends

Absent treatment, treated and control groups would have followed **parallel** outcome trends. Cannot be tested directly (no counterfactual), but supporting evidence:

- **Visual inspection** of pre-treatment trends.
- **Placebo event-study**: pre-treatment leads (`τ_{k<0}`) should be small and non-significant.
- **Triple-differences** or **synthetic control** to relax the assumption.

## Standard errors

- **Cluster-robust at the unit level** — errors are correlated within unit over time (see `sandwich-robust-se`).
- Rule of thumb: `G ≥ 40` clusters for cluster-robust SEs to work; below that, use **wild-cluster bootstrap** (Cameron-Gelbach-Miller 2008).

## Run

```
python techniques/diff-in-diff/python/diff_in_diff.py
Rscript techniques/diff-in-diff/r/diff_in_diff.R
```

**Refs:** Card, D. & Krueger, A.B. "Minimum wages and employment: a case study of the fast-food industry in New Jersey and Pennsylvania." *AER* 84(4), 772–793, 1994; Goodman-Bacon, A. "Difference-in-differences with variation in treatment timing." *J. Econometrics* 225(2), 254–277, 2021; Callaway, B. & Sant'Anna, P.H.C. "Difference-in-differences with multiple time periods." *J. Econometrics* 225(2), 200–230, 2021.

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
