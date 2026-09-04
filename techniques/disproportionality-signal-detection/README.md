# Disproportionality Analysis (Reference §43.1)

Bate & Evans (2009), DuMouchel (1999). Signal-detection method for
spontaneous adverse-drug-reaction reports (FAERS, VigiBase,
EudraVigilance). Build a 2×2 for each (drug, event) pair and test
whether the reporting proportion is disproportionately high.

## Statistics

| Metric | Formula |
|---|---|
| **PRR** (proportional reporting ratio) | `(a/(a+b)) / (c/(c+d))` |
| **ROR** (reporting odds ratio) | `(a·d) / (b·c)` |
| **IC** (information component, BCPNN) | `log₂((a + 0.5) / (E + 0.5))`, `E = (a+b)(a+c)/N` |

EMA signal criteria: `PRR ≥ 2 AND a ≥ 3 AND χ² ≥ 4`.

## When to use

- **Post-marketing pharmacovigilance** — data-mining large SRSs.
- **Vaccine safety** — VAERS review.
- **Early-warning triage** — prioritise events for regulatory
  review.

## When NOT to use

- **Causal inference** — SRS data suffer from reporting bias, no
  denominator; signals are hypothesis-generating only.
- **Rare-event confirmation** — small counts inflate PRR /
  ROR; use IC / MGPS which shrink toward the null.

## Files

- `python/disproportionality_signal_detection.py` — PRR, ROR, IC,
  χ² on a 2×2 (custom). Demo (3 drug-event pairs): drug1
  (a=60, b=400) → PRR 26.2, χ² 917, IC +3.9 → signal; drug2 (a=5)
  → PRR 4.0, χ² 10.1, IC +1.6 → signal; drug3 (a=3, rare) → PRR
  29.2 but a-count borderline.
- `r/disproportionality_signal_detection.R` —
  `PhViD::bcpnn`/`PRR`/`ROR`/`MGPS`, `pvLRT` (R); `vigipy` +
  custom (Python).

## Assumptions & caveats

- **Reporting bias** dominates — Weber effect, notoriety bias, and
  competition bias distort all metrics.
- **Multiple testing** — millions of (drug, event) cells; MGPS
  (DuMouchel 1999) uses empirical-Bayes shrinkage.
- **No denominator** — cannot compute rates or absolute risk from
  SRSs.
- **Missing / duplicate reports** — pre-clean before signal
  detection.

## Related in this repo

- `bayesian-glms` — Bayesian shrinkage relatives.
- `multiple-testing-corrections` — corrections across cells.
- `benefit-risk-mcda`, `sccs-self-controlled` — downstream
  regulatory workflows.

## Run

```
python techniques/disproportionality-signal-detection/python/disproportionality_signal_detection.py
Rscript techniques/disproportionality-signal-detection/r/disproportionality_signal_detection.R
```

**Refs:** Bate, A. & Evans, S.J.W. "Quantitative signal detection using spontaneous ADR reporting." *Pharmacoepidemiology and Drug Safety*, 2009; DuMouchel, W. "Bayesian data mining in large frequency tables, with an application to the FDA spontaneous reporting system." *The American Statistician*, 1999.

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
