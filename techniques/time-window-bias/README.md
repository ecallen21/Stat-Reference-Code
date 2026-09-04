# Time-Window Bias (Reference §43.15)

Suissa & Dell'Aniello (2012). In case-control studies, different
observation windows for cases and controls create **spurious**
exposure-outcome associations even when the exposure has no effect.

The classical failure: "ever use" ascertained from the entire
available history for cases (dating back years) but only from a
short window before the index date for controls. Cases mechanically
show more "ever use" — an artefact of the design.

## Fix

Use a **common look-back window** for cases and controls —
equivalent duration + calendar-aligned. Sensitivity analyses over
several window lengths.

## When it appears

- **Case-control studies** on chronic-drug exposures.
- **Cohort studies with variable follow-up** where prevalence is
  reported at time zero.
- **Registry linkage** with cases followed longer than controls.

## Files

- `python/time_window_bias.py` — simulate identical 10 %/year
  prescribing across all patients but observe cases for ~5.8 yr
  and controls for 1 yr. Demo: **naive OR = 5.31**; common
  1-year window → **OR = 0.91** — bias vanishes.
- `r/time_window_bias.R` — `EHR`, `acs` (R); `zepid`, custom
  (Python).

## Assumptions & caveats

- **Report the window** used and vary it as a sensitivity
  analysis.
- **Calendar-time alignment** — a window earlier for cases may
  span a different prescribing era.
- **Truncated history** for controls — if the database only
  covers a short window, cases must be similarly truncated.
- **Related biases** — immortal-time (see companion technique),
  protopathic, reverse causation.

## Related in this repo

- `immortal-time-bias`, `confounding-by-indication`,
  `new-user-active-comparator`, `target-trial-emulation` — the
  bias-mitigation toolbox.

## Run

```
python techniques/time-window-bias/python/time_window_bias.py
Rscript techniques/time-window-bias/r/time_window_bias.R
```

**Refs:** Suissa, S. & Dell'Aniello, S. "Time-window bias in case-control studies: statins and lung cancer." *Epidemiology*, 2012; Schneeweiss, S. & Avorn, J. "A review of uses of health care utilization databases for epidemiologic research on therapeutics." *Journal of Clinical Epidemiology*, 2005.

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
