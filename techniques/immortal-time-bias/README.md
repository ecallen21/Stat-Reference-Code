# Immortal-Time Bias (Reference §38.25)

Suissa (2008), Lévesque et al. (2010). **Immortal time** = person-time
during follow-up in which the outcome cannot occur by definition of
the exposure classification. Endemic in pharmacoepidemiology and
routinely produces spurious "protective" treatment effects.

## The prototypical failure

Define "treated" = "ever received drug X during follow-up". A
patient who dies before their first prescription is (mis-)classified
as **untreated**. The time from cohort entry to first prescription is
guaranteed-alive time (immortal) attributed to the treated group,
which artificially inflates their survival.

## Standard fixes

- **Time-varying exposure** — a subject contributes person-time to
  "untreated" before first prescription, "treated" after. Use
  `survival::tmerge` in R or `CoxTimeVaryingFitter` in lifelines.
- **Target-trial emulation** — align time zero for the emulated
  trial with treatment initiation; assess eligibility at baseline
  only.
- **Active comparator / new user (ACNU)** — restrict to new
  initiators of one of two drugs, indexed at first prescription.

## When it matters

- **Pharmacoepi cohort studies** — any drug ↔ mortality question.
- **Registry studies** — treatment status ascertained during
  follow-up.
- **Observational device / procedure studies** — same failure
  mode.

## Files

- `python/immortal_time_bias.py` — simulate a cohort where
  treatment truly has **no effect** on survival, then fit Cox two
  ways. Demo (n=2000, median survival ≈2 yr, τ_treat ≈6 mo):
  **Naive baseline-fixed HR = 0.122** (huge spurious protective
  effect); **time-varying Cox HR = 1.13** (unbiased around the truth
  of 1.0).
- `r/immortal_time_bias.R` — `survival::tmerge`,
  `TrialEmulation` (R); `lifelines.CoxTimeVaryingFitter`, `zepid`
  (Python).

## Assumptions & caveats

- **Requires date of exposure start**, not just "ever exposed".
- **Time-varying analysis needs the correct assumption about
  exposure at each event time** — misclassified start dates
  reintroduce bias.
- **Landmark analysis** is a partial fix (restrict cohort to
  survivors past a landmark) but wastes data and cannot recover the
  effect on early events.
- **Prevalent-user bias** is a cousin — restrict to new users to
  avoid it as well.

## Related in this repo

- `cox-ph`, `time-varying-covariates` — the modelling machinery.
- `target-trial-emulation` (if present) — design-level fix.
- `active-comparator-new-user` (if present) — the ACNU template.

## Run

```
python techniques/immortal-time-bias/python/immortal_time_bias.py
Rscript techniques/immortal-time-bias/r/immortal_time_bias.R
```

**Refs:** Suissa, S. "Immortal time bias in pharmacoepidemiology." *American Journal of Epidemiology*, 2008; Lévesque, L.E., Hanley, J.A., Kezouh, A., & Suissa, S. "Problem of immortal time bias in cohort studies: example using statins for preventing progression of diabetes." *BMJ*, 2010; Lund, J.L., Richardson, D.B., & Stürmer, T. "The active comparator, new user study design in pharmacoepidemiology." *Pharmacoepidemiology and Drug Safety*, 2015.

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
