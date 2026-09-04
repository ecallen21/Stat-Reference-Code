# Agreement Beyond Kappa (Reference §38.19)

Gwet (2014), Krippendorff (2019). Cohen's κ suffers two documented
paradoxes:

1. High observed agreement gives near-zero κ when marginals are
   very unbalanced (prevalence paradox).
2. Symmetric vs asymmetric marginal imbalance gives very different
   κ for identical observed agreement (bias paradox).

## Alternatives

- **PABAK** (Byrt-Bishop-Carlin 1993) — prevalence-adjusted bias-
  adjusted κ; `PABAK = (K · P_a − 1) / (K − 1)` for K categories.
- **Gwet's AC1** (2008) — chance correction uses the average
  marginal, resistant to both paradoxes.
- **Krippendorff's α** — any number of raters, any measurement
  level (nominal, ordinal, interval, ratio); handles missing data
  by design.

## When to use

- **Rare-outcome coding** — infection-control chart reviews,
  radiologist "positive" reads for uncommon findings.
- **Multi-rater / multi-scale reliability** — content analysis
  studies where κ is a poor fit.
- **Missing-data-ubiquitous designs** — Krippendorff's α is the
  standard.

## When NOT to use

- **Balanced 2-rater, 2-category, moderate agreement** — Cohen's
  κ is fine.
- **Ordinal / interval scales** with a natural distance — use a
  weighted κ or ordinal Krippendorff's α, not the nominal form.

## Files

- `python/agreement_beyond_kappa.py` — Cohen κ, PABAK, Gwet AC1,
  nominal Krippendorff α (multi-rater, missing OK). Demo of the
  paradox: for `[[95, 2], [3, 0]]` (95 % agreement, extreme
  prevalence), **Cohen κ = −0.025** vs **PABAK = 0.900**,
  **AC1 = 0.947**. Multi-rater α on 80 items × 5 raters at 90 %
  accuracy: **α = 0.77**.
- `r/agreement_beyond_kappa.R` — `irrCAC::gwet.ac1`,
  `irr::kripp.alpha`, `psych::cohen.kappa` (R); `krippendorff`,
  `sklearn.metrics.cohen_kappa_score` (Python).

## Assumptions & caveats

- **Prevalence dependence** — always report the marginal
  distributions alongside any chance-corrected coefficient.
- **Scale of measurement** — the nominal form of α assumes
  categorical values with no distance; use ordinal / interval
  distances for those scales.
- **Bootstrap CIs recommended** — closed-form SEs for AC1 and α
  are available (Gwet 2014) but bootstrap is more robust in small
  samples.
- **Missing at random** — Krippendorff's α tolerates missing
  ratings under MAR; systematic missingness biases the estimate.

## Related in this repo

- `cohens-kappa` (baseline) — see the classical method.
- `icc` (if present) — intraclass correlation for continuous
  ratings.
- `weighted-kappa` (if present) — ordinal distance corrections.

## Run

```
python techniques/agreement-beyond-kappa/python/agreement_beyond_kappa.py
Rscript techniques/agreement-beyond-kappa/r/agreement_beyond_kappa.R
```

**Refs:** Gwet, K.L. *Handbook of Inter-Rater Reliability*, 4th ed., Advanced Analytics, 2014; Krippendorff, K. *Content Analysis: An Introduction to Its Methodology*, 4th ed., SAGE, 2019; Byrt, T., Bishop, J., & Carlin, J.B. "Bias, prevalence, and kappa." *Journal of Clinical Epidemiology*, 1993.

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
