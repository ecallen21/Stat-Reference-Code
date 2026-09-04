# Target Trial Emulation (Reference §43.11)

Hernán & Robins (2016), Dickerman et al. (2019). Framework for
comparative-effectiveness questions in observational data: first
**write the protocol** of the ideal randomised trial you would run
if resources allowed, then **emulate each element** with claims /
EHR data.

## Seven protocol elements

1. **Eligibility** criteria applied at time zero.
2. **Treatment strategies** (usually new-user, active-comparator).
3. **Assignment** procedure (mimicked via propensity score or
   grace-period design).
4. **Follow-up** — start and end (time zero to outcome / censoring).
5. **Outcome** definition + measurement.
6. **Causal contrast** — per-protocol vs intention-to-treat.
7. **Statistical analysis** — IPTW, g-formula, g-estimation.

## When to use

- **Comparative effectiveness** on claims / EHR when RCTs are
  unavailable, unethical, or already conducted only in narrow
  populations.
- **Regulatory** post-marketing effectiveness / safety questions.

## When NOT to use

- **Unmeasured strong confounders** — no PS method rescues the
  study.
- **Truly randomisable questions** with modest cost — run the RCT.

## Files

- `python/target_trial_emulation.py` — new-user, active-comparator
  IPTW analysis. Demo (n=3000, 2 confounders, true ATE=0.30):
  naive difference **0.42** (confounded); target-trial-emulated
  IPTW ATE **0.30** — exact truth recovery.
- `r/target_trial_emulation.R` — `TrialEmulation`, `CohortMethod`,
  `Cyclops`, `FeatureExtraction` (OHDSI) (R); `zepid`,
  `TrialEmulation` via rpy2 (Python).

## Assumptions & caveats

- **Time-zero alignment** — the single most common failure mode is
  misalignment of eligibility, treatment start, and follow-up
  start.
- **Prevalent-user bias** — restrict to new users of the study
  drug and its comparator.
- **Immortal-time bias** — grace periods or clone-and-censor
  designs avoid it (see `immortal-time-bias`).
- **Publication protocol** first — pre-register the target trial
  as if it were an RCT.

## Related in this repo

- `hdps-high-dim-propensity` — automated covariate selection
  companion.
- `immortal-time-bias`, `new-user-active-comparator` — design-level
  pitfalls / fixes.
- `negative-outcome-controls` — residual-confounding sensitivity
  analyses.

## Run

```
python techniques/target-trial-emulation/python/target_trial_emulation.py
Rscript techniques/target-trial-emulation/r/target_trial_emulation.R
```

**Refs:** Hernán, M.A. & Robins, J.M. "Using big data to emulate a target trial when a randomized trial is not available." *American Journal of Epidemiology*, 2016; Dickerman, B.A., García-Albéniz, X., Logan, R.W. et al. "Avoidable flaws in observational analyses: an application to statins and cancer." *Nature Medicine*, 2019.

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
