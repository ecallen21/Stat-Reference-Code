# Trim-and-Fill (Reference §22.4)

Duval & Tweedie (2000). Nonparametric adjustment for publication
bias in meta-analysis. If small negative studies are missing from a
funnel plot (asymmetric funnel):

1. **Trim** — remove the most extreme small-positive studies
   until the funnel is symmetric.
2. **Estimate `k_0`** — Duval-Tweedie L₀ / R₀ estimator of missing
   studies.
3. **Fill** — add `k_0` imputed studies as mirror reflections of
   the trimmed extremes.
4. **Re-pool** — random-effects meta-analysis on the augmented set.

## When to use

- **Symmetric funnel-plot** suggests bias — pair with Egger's
  test.
- **Sensitivity analysis** — report adjusted vs unadjusted pooled
  effect.

## When NOT to use

- **Heterogeneity, not bias** — asymmetry can also reflect a real
  effect × study-size interaction; not always publication bias.
- **Very small K** (< 10) — L₀ estimator is unstable.
- **Confirmatory conclusion** — trim-fill is exploratory; never
  the primary analysis.

## Files

- `python/trim_fill.py` — L₀ estimator + mirror imputation (single
  iteration, standard). Demo (K=20, 2 studies suppressed due to
  small-effect-with-large-SE mask): naive pooled **+0.378**,
  trim-fill augmented **+0.355** with 2 filled studies.
- `r/trim_fill.R` — `metafor::trimfill`, `meta::trimfill`,
  `metabias` (R); `pymare` + custom (Python).

## Assumptions & caveats

- **Symmetric funnel** under no bias — an assumption violated by
  moderator effects.
- **L₀ vs R₀** — different estimators; use the one that
  `metafor::trimfill` defaults to (L₀ or R₀ depending on side).
- **Report as sensitivity** — Sterne et al. 2001 caution against
  trim-fill as primary evidence.
- **Egger's / Begg's tests** are complementary bias diagnostics.

## Related in this repo

- `meta-analysis`, `meta-regression` — parent methods.
- `publication-bias` (if present) — companion diagnostics.

## Run

```
python techniques/trim-fill/python/trim_fill.py
Rscript techniques/trim-fill/r/trim_fill.R
```

**Refs:** Duval, S. & Tweedie, R. "A nonparametric 'trim and fill' method of accounting for publication bias in meta-analysis." *JASA*, 2000; Sterne, J.A.C., Egger, M., & Smith, G.D. "Investigating and dealing with publication and other biases in meta-analysis." *BMJ*, 2001.

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
