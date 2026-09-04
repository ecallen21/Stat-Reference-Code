# Drug Utilization & Adherence (Reference §43.7)

WHO Collaborating Centre for Drug Statistics Methodology (2024),
Andrade et al. (2006). Standard adherence and utilisation metrics
computed from pharmacy dispensing records.

## Core metrics

- **MPR** (Medication Possession Ratio) = `Σ days_supplied /
  observation_days`; may exceed 1 with overlap.
- **PDC** (Proportion of Days Covered) = `days with any supply on
  hand / observation_days`; capped at 1. CMS Star-rating standard.
- **Persistence** — days from initiation to the first gap > a
  threshold (usually 30 or 60 days).
- **DDD** (WHO ATC/DDD) — total drug quantity ÷ defined daily
  dose = equivalent days of therapy on a standardised dose.

## When to use

- **Adherence research** — real-world persistence of chronic-
  disease medications.
- **Health-services research** — Medicare Star Ratings, employer
  formulary benchmarks.
- **Comparative effectiveness** — adherence-adjusted analyses.

## When NOT to use

- **Non-refill adherence** (inhaler technique, dosing correctness)
  — dispensing metrics do not measure it.
- **In-hospital medications** — pharmacy claims miss these.

## Files

- `python/drug_utilization_adherence.py` — MPR / PDC / persistence
  from a list of `(start_day, days_supplied)` fills. Demo:
  perfect refills → MPR = PDC = 0.986; sporadic (overlaps + gaps)
  → MPR 0.49, PDC 0.48, persistence 55 days; DDD example converts
  1200 mg total at 20 mg DDD → 60 equivalent days.
- `r/drug_utilization_adherence.R` — `AdhereR::CMA` +
  `CMA_per_episode`, `survey` (R); `pandas` custom, `lifelines`
  (Python).

## Assumptions & caveats

- **Fill = consumption** — the fundamental adherence assumption;
  refills are a proxy.
- **Grace periods** and stockpiling — PDC handles overlap; MPR does
  not.
- **Multi-drug regimens** — separate PDC per drug or a combined
  PDC across the regimen; report method.
- **Time zero** — first fill vs first day of coverage; be explicit.

## Related in this repo

- `time-window-bias`, `immortal-time-bias` — related design
  pitfalls.
- `landmark-analysis` — for adherence-adjusted outcome analyses.

## Run

```
python techniques/drug-utilization-adherence/python/drug_utilization_adherence.py
Rscript techniques/drug-utilization-adherence/r/drug_utilization_adherence.R
```

**Refs:** WHO Collaborating Centre for Drug Statistics Methodology. *Guidelines for ATC Classification and DDD Assignment*, WHO, 2024; Andrade, S.E., Kahler, K.H., Frech, F., & Chan, K.A. "Methods for evaluation of medication adherence and persistence using automated databases." *Pharmacoepidemiology and Drug Safety*, 2006.

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
