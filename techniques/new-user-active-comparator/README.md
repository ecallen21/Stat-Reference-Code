# New-User, Active-Comparator (ACNU) Design (Reference §43.14)

Lund et al. (2015), Ray (2003). Standard pharmacoepi design that
combines two pillars:

- **New user (incident user)** — include only patients initiating
  the study drug for the first time. Requires a **washout period**
  (usually 6-12 months) of no prior use before the qualifying
  first fill.
- **Active comparator** — the comparison group is a **different
  drug indicated for the same condition**, not "no drug". Aligns
  indication and calendar time.

Together they eliminate **prevalent-user bias** (early
discontinuers and adverse-reaction dropouts are already gone from
the "current user" pool) and reduce **confounding by indication**.

## When to use

- **All pharmacoepi effectiveness / safety studies** on chronic
  drugs — this is the default modern design.
- **Multi-database / OHDSI** analyses.

## When NOT to use

- **Short-term acute drugs** — the new-user restriction may
  eliminate too much of the eligible cohort.
- **When no reasonable active comparator exists** — negative-
  control outcomes or falsification tests can partially substitute.

## Files

- `python/new_user_active_comparator.py` — build new-user cohort
  with configurable washout; restrict to two active comparators.
  Demo (500 patients, 30 % prevalent users): with **study
  start = day 400 + 365-day washout**, cohort drops to **440**;
  drug A vs drug B distribution 266 / 174.
- `r/new_user_active_comparator.R` — `CohortMethod` (OHDSI:
  `createStudyPopulation`), `MatchIt`, `WeightIt`, `cobalt` (R);
  `zepid`, custom `pandas` (Python).

## Assumptions & caveats

- **Washout length** — 6 months minimum for "new user"; 12 months
  is more conservative. Report the choice and its sensitivity.
- **Same indication** — verify (chart review / diagnosis codes)
  that both study drugs are truly used for the same condition.
- **Grace period** for treatment switching — pre-specify.
- **Follow-up start = time zero** — never before, or immortal-time
  bias returns.

## Related in this repo

- `immortal-time-bias`, `confounding-by-indication`,
  `target-trial-emulation` — the design-level toolbox.
- `hdps-high-dim-propensity`, `propensity-score-matching` —
  covariate-adjustment layers on top of ACNU.

## Run

```
python techniques/new-user-active-comparator/python/new_user_active_comparator.py
Rscript techniques/new-user-active-comparator/r/new_user_active_comparator.R
```

**Refs:** Lund, J.L., Richardson, D.B., & Stürmer, T. "The active comparator, new user study design in pharmacoepidemiology: historical foundations and contemporary application." *Current Epidemiology Reports*, 2015; Ray, W.A. "Evaluating medication effects outside of clinical trials: new-user designs." *American Journal of Epidemiology*, 2003.

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
