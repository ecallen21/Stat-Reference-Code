# Event-Study Design (Reference §35.11)

Regress the outcome on **lead and lag treatment dummies**:

```
y_it = α_i + γ_t + Σ_{k ∈ K} β_k · 𝟙(t − g_i = k) + ε_it,
```

with `k = −1` omitted as the reference. Pre-treatment coefficients
`{β_{-3}, β_{-2}}` test **parallel trends**; post-treatment
`{β_0, β_1, …}` trace **dynamic effects**.

## Modern-econometrics warning

With **staggered adoption AND heterogeneous effects**, two-way-FE
event studies are contaminated (Sun-Abraham 2021, Goodman-Bacon 2021):
pre-period coefficients need not be zero even under parallel trends,
and post-period coefficients mix effects across cohorts. Use
**Sun-Abraham interaction-weighted estimator** or **Callaway-Sant'Anna
staggered DiD** (see `staggered-did`).

## When to use

- **Single-cohort treatment** — the classical version is fine.
- **Diagnosing parallel-trends assumption** — plot pre-coefs.
- **Ceremony / reporting** — event-study plots are a standard
  visual.

## When NOT to use (unmodified)

- **Multi-cohort staggered adoption** — use robust variants.
- **Anticipation effects** — leads absorb them; include enough lead
  dummies.

## Files

- `python/event_study.py` — from-scratch OLS event study with unit +
  time FE + event-time dummies. Demo with **two treated cohorts**
  (t = 4 and t = 7) reveals the TWFE staggered pathology: pre-period
  coefs at ≈ −1 (not 0) and post-period coefs offset by ≈ −1 from
  truth — a live demonstration of the "forbidden comparison"
  problem.
- `r/event_study.R` — `fixest::feols i(event, ref = −1)`,
  `fixest::sunab` (R); `pyfixest` (Python).

## Assumptions & caveats

- **Reference period** — `k = −1` is standard; omitting it makes the
  remaining coefs relative to the period just before treatment.
- **Endpoints binning** — extreme leads / lags are usually binned
  ("k = −3 or earlier") to preserve degrees of freedom.
- **Two-way FE pathology** — heterogeneous effects across cohorts
  contaminate; use `staggered-did`.
- **Cluster-robust SEs** — cluster by unit; wild-bootstrap for small
  cluster counts.
- **Test of pre-trends** — joint F test on `β_{k < 0} = 0`; some
  authors argue this is under-powered (Roth 2022).

## Related in this repo

- `diff-in-diff`, `staggered-did`, `synthetic-did` — sibling designs
  (this batch).
- `fixed-effects-panel`, `hausman-test` — the panel machinery.
- `newey-west-hac` — for time-serial errors in event windows.

## Run

```
python techniques/event-study/python/event_study.py
Rscript techniques/event-study/r/event_study.R
```

**Refs:** Borusyak, K., Jaravel, X. & Spiess, J. "Revisiting event study designs: robust and efficient estimation." *Review of Economic Studies*, 2024; Sun, L. & Abraham, S. "Estimating dynamic treatment effects in event studies with heterogeneous treatment effects." *Journal of Econometrics*, 2021; Roth, J. "Pre-test with caution: event-study estimates after testing for parallel trends." *AER: Insights*, 2022.

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
