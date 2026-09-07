# Case-Crossover Study (Reference §15.36)

Maclure (1991). Each case serves as **its own control** — exposure
during a short "hazard period" immediately before the acute event is
compared with exposure during "control periods" from the same person
at different times. All time-invariant confounders (genetics, chronic
disease, SES) cancel within subject.

## Analytic form

For 1 hazard period + M control periods per case, conditional
logistic regression stratified by subject gives the OR:

    (1:1)  OR = f_{10} / f_{01}                (McNemar-style)
    (1:M)  OR = Mantel-Haenszel across strata

## Files

- `python/case_crossover.py` — McNemar and Mantel-Haenszel estimators
  from scratch. Demo (n=800 cases, true OR 2.5, exposure prevalence
  0.15): 1:1 OR = 2.81, 1:3 OR = 2.32.
- `r/case_crossover.R` — `survival::clogit`, `Epi::clogistic` (R);
  `statsmodels.discrete.conditional_models.ConditionalLogit` (Python).

## When to use

- **Acute events with transient exposures** — MI after strenuous
  exertion, asthma attack after allergen exposure, injury after
  ambien.
- **When time-invariant confounders dominate** — the design nullifies
  them by construction.
- **Rare exposures / rare outcomes** — no need for a large control
  cohort.

## When NOT to use

- **Cumulative / chronic exposures** — the design assumes the
  exposure switches on/off; steady-state exposures make hazard and
  control periods identical.
- **Trends over time** — if exposure prevalence trends up/down,
  compare with a **case-time-control** design.
- **Long induction period** — the hazard window must be biologically
  plausible; a mismatch dilutes the OR.

## Assumptions & caveats

- **Exchangeability of hazard vs control periods** — the person must
  be similar in every way except exposure. Seasonality and time
  trends violate this; use time-stratified sampling.
- **No carry-over** — control periods precede the hazard period far
  enough that exposure effects have washed out.
- **Effect must be transient** — if a chronic exposure raises baseline
  risk permanently, the design cannot detect it.
- **Autocorrelation of exposure** — if exposure is highly persistent
  within subject, control periods leak into the hazard period.

## Related in this repo

- `sccs-self-controlled` — self-controlled case series (an
  epidemiologic cousin using the full observation window).
- `exposure-crossover`, `prescription-sequence-symmetry` — related
  self-controlled pharmacoepi designs.
- `cox-time-varying` — time-varying-covariate cohort alternative.

## Run

```
python techniques/case-crossover/python/case_crossover.py
Rscript techniques/case-crossover/r/case_crossover.R
```

**Refs:** Maclure, M. "The case-crossover design: a method for studying transient effects on the risk of acute events." *American Journal of Epidemiology*, 133(2): 144-153, 1991; Mittleman, M.A. & Mostofsky, E. "Exchangeability in the case-crossover design." *IJE*, 43(5): 1645-1655, 2014.

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
