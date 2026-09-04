# Confounding by Indication + Protopathic Bias (Reference §43.12)

Salas-Hofman-Stricker (1999), Horwitz & Feinstein (1980). Three
pharmacoepi-specific biases:

- **Confounding by indication** — sicker patients receive
  treatment, creating spurious treatment-outcome association.
- **Channelling bias** — prescribers selectively route certain
  drugs to certain risk groups (e.g., safer drugs to frail
  patients).
- **Protopathic bias** — symptoms of an as-yet-undiagnosed disease
  trigger a prescription that is then mistakenly implicated as the
  cause.

## Fixes

- **New-user, active-comparator design** — restrict to new
  initiators of one of two drugs indicated for the same condition
  (see `new-user-active-comparator`).
- **Adjust for indication severity** — biomarker, disease stage,
  score.
- **Lag exposure window** — set a washout of weeks-to-months
  before follow-up to attenuate protopathic bias.
- **Instrumental variables** where a valid instrument exists.

## Files

- `python/confounding_by_indication.py` — naive vs severity-
  adjusted regression + a rough active-comparator subset. Demo
  (n=4000, unmeasured severity strongly predicts treatment,
  true drug effect = −0.3): naive difference **+0.76 (biased)**;
  severity-adjusted **−0.30 (unbiased)**; active-comparator
  subset **−0.76** (illustrative — real ACNU designs perform
  much better than this loose construction).
- `r/confounding_by_indication.R` — `MatchIt`, `WeightIt`,
  `cobalt`, `CohortMethod` (OHDSI) (R); `causalinference`,
  `zepid`, `dowhy` (Python).

## Assumptions & caveats

- **Measured confounders only** — regression / matching / IPTW
  cannot fix unmeasured confounding by indication.
- **Active-comparator validity** — requires both drugs to be truly
  indicated for the same condition and time period.
- **Protopathic-bias windows** are not a free lunch — a long lag
  may drop real early-onset events.
- **Report the design choice explicitly** — the target-trial
  framework forces this discipline.

## Related in this repo

- `new-user-active-comparator`, `target-trial-emulation`,
  `immortal-time-bias` — design-level fixes.
- `propensity-score-matching`, `hdps-high-dim-propensity` —
  covariate adjustment.

## Run

```
python techniques/confounding-by-indication/python/confounding_by_indication.py
Rscript techniques/confounding-by-indication/r/confounding_by_indication.R
```

**Refs:** Salas, M., Hofman, A., & Stricker, B.H.C. "Confounding by indication: an example of variation in the use of epidemiologic terminology." *American Journal of Epidemiology*, 1999; Horwitz, R.I. & Feinstein, A.R. "The problem of 'protopathic bias' in case-control studies." *American Journal of Medicine*, 1980.

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
