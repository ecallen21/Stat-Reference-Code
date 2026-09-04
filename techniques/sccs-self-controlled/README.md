# Self-Controlled Case Series (Reference §43.2)

Farrington (1995), Petersen-Douglas-Whitaker (2016). **Only cases**
(people with the outcome) are analysed. Each case's follow-up is
split into **risk** (post-exposure) and **baseline** windows. A
conditional Poisson likelihood removes all time-invariant
confounders (genetics, chronic disease, socio-economic status).

## Model

```
N_ij ~ Poisson(μ_ij),  log μ_ij = φ_i + log e_ij + β · x_ij
```

Conditioning on `Σ_j N_ij` eliminates `φ_i` (individual baseline);
`β` is the log-IRR of risk vs baseline window.

## When to use

- **Vaccine safety** — MMR-autism debate was resolved via SCCS.
- **Transient exposures** — antibiotics, opioids, biologics with a
  defined risk window.
- **Rare outcomes** — no need for a comparison cohort.

## When NOT to use

- **Chronic / permanent exposures** — no baseline to compare.
- **Outcome affects future exposure** (event-dependent exposure) —
  violates the SCCS assumption.
- **Non-recurrent fatal outcomes** — extensions (Farrington 2011)
  handle these but standard SCCS is biased.

## Files

- `python/sccs_self_controlled.py` — 2-window conditional-Poisson
  IRR with Wald CI (custom). Demo (500 cases, baseline rate 0.5 /
  py, risk rate 1.5 / py, IRR truth 3.0): **estimated IRR 2.61
  (95 % CI 2.01, 3.38)** — CI covers truth.
- `r/sccs_self_controlled.R` — `SCCS::standardsccs`/`semisccs`,
  `gnm`, `SelfControlledCaseSeries` (OHDSI) (R); custom (Python).

## Assumptions & caveats

- **Independent recurrences** — multiple events per person are
  assumed independent of prior events.
- **Event-independent exposure** — the outcome does not affect
  future exposure probability.
- **Event-independent observation period** — censoring is
  independent of the event.
- **Age effect** — include as a stratified nuisance term for
  developmental / age-dependent outcomes.
- **Time-varying confounders** — SCCS does not adjust for these;
  include as covariates or extend to season-adjusted SCCS.

## Related in this repo

- `case-crossover` (if present) — companion within-person design.
- `immortal-time-bias`, `new-user-active-comparator` — cohort-
  design cousins.
- `disproportionality-signal-detection` — SRS-based signal.

## Run

```
python techniques/sccs-self-controlled/python/sccs_self_controlled.py
Rscript techniques/sccs-self-controlled/r/sccs_self_controlled.R
```

**Refs:** Farrington, C.P. "Relative incidence estimation from case series for vaccine safety evaluation." *Biometrics*, 1995; Petersen, I., Douglas, I., & Whitaker, H. "Self controlled case series methods: an alternative to standard epidemiological study designs." *BMJ*, 2016.

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
