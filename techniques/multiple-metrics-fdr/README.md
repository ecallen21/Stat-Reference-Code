# Multiple Metrics + FDR (Reference §44.5)

Kohavi-Tang-Xu (2020 ch 17), Benjamini & Hochberg (1995). A single
A/B test typically tracks dozens of metrics: primary, secondary,
guardrail. Corrections:

- **Bonferroni** — `α / m`; FWER control, very conservative.
- **Benjamini-Hochberg (BH)** — FDR control; less conservative,
  standard for large metric families.
- **Hierarchical** — test primary at `α₁`; only if primary passes,
  test secondaries at `α₂` each. Simple, transparent, widely used.

## When to use

- **Any A/B report** with more than one metric — always pre-specify
  the family and the correction procedure.
- **Automated experimentation platforms** — use BH-FDR across the
  metric suite by default.

## When NOT to use

- **A single, pre-specified primary metric** — no correction
  needed; secondaries are exploratory.
- **Correlated metrics** (dozens of tightly-related count metrics)
  — BH is valid but conservative; consider Storey q-values or
  hierarchical.

## Files

- `python/multiple_metrics_fdr.py` — BH-FDR, Bonferroni,
  hierarchical (custom). Demo (m=20, 5 true, 15 null): uncorrected
  rejects **9** (TP=5, FP=4); Bonferroni **0**; **BH rejects 2
  (TP=1, FP=1)** — dramatically tighter FDR at some power cost.
- `r/multiple_metrics_fdr.R` — `stats::p.adjust`, `qvalue`,
  `mutoss` (R); `statsmodels.stats.multitest.multipletests`
  (Python).

## Assumptions & caveats

- **Pre-specify** the correction family before looking at any
  metric.
- **Correlated tests** — BH is valid under positive-dependence
  (PRDS); BY is safer for arbitrary dependence.
- **Primary / secondary hierarchy** must be pre-registered too.
- **Report the uncorrected p AND the correction rule** — audit
  trail matters.

## Related in this repo

- `multiple-testing-corrections`, `false-discovery-rate` (if
  present) — general primers.
- `guardrail-monitoring` — hierarchical procedure in production.

## Run

```
python techniques/multiple-metrics-fdr/python/multiple_metrics_fdr.py
Rscript techniques/multiple-metrics-fdr/r/multiple_metrics_fdr.R
```

**Refs:** Kohavi, R., Tang, D., & Xu, Y. *Trustworthy Online Controlled Experiments*, Cambridge University Press, 2020 (ch 17); Benjamini, Y. & Hochberg, Y. "Controlling the false discovery rate: a practical and powerful approach to multiple testing." *JRSS-B*, 1995.

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
