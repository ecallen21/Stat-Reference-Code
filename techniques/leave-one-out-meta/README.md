# Leave-One-Out Meta-Analysis (Reference §47.273)

Standard sensitivity analysis: refit the meta-analysis leaving
out each study in turn, showing how the pooled estimate and
its CI change. Flags studies that are individually influential
or that drive the overall conclusion.

## Files

- `python/leave_one_out_meta.py` — DL-based L1O table with
  flagging of studies whose removal shifts the pooled estimate
  by more than 2 SE. Demo: 8 well-behaved studies + 1 outlier
  (y=1.5, v=0.02). Full pool = 0.505; omitting the outlier
  shifts to 0.395 (−0.109). The outlier is flagged
  "INFLUENTIAL".
- `r/leave_one_out_meta.R` — `metafor::leave1out`,
  `meta::metainf`, `dmetar::InfluenceAnalysis` (R); PythonMeta,
  from-scratch (Python).

## When to use

- **Any meta-analysis** — L1O is the workhorse sensitivity
  analysis.
- **Alongside publication bias tests** — remove suspect
  small studies to check robustness.
- **Suspected outlier or unusual study** — L1O quantifies its
  leverage on the pooled estimate.

## When NOT to use

- **Very small K (< 5)** — with few studies, dropping one
  changes everything; L1O is uninformative.
- **Substituting for pre-specified subgroup analysis** —
  L1O is post-hoc, not a formal hypothesis test.
- **Instead of Baujat / influence plots** — L1O misses joint
  influence; consider Baujat's Q vs contribution plot too.

## Assumptions & caveats

- **DL vs REML** — L1O tables assume the same τ² estimator
  as the full analysis; report both.
- **Ordering matters for pattern-detection** — sort by
  publication year, sample size, or effect magnitude to see
  systematic patterns.
- **Combined with cumulative meta-analysis** — L1O and
  cumulative plots complement each other.
- **Formal flag** — Cochrane recommends flagging studies
  whose removal changes the conclusion regarding significance,
  not just the point estimate.

## Related in this repo

- `dersimonian-laird-random-effects`,
  `hartung-knapp-sidik-jonkman` — the base analyses.
- `cumulative-meta-analysis` — the "add-one-at-a-time" cousin.
- `bootstrap-optimism-correction` — a different resampling
  sensitivity method.

## Run

```
python techniques/leave-one-out-meta/python/leave_one_out_meta.py
Rscript techniques/leave-one-out-meta/r/leave_one_out_meta.R
```

**Refs:** Widely used sensitivity technique; formally
introduced in early Cochrane Handbook editions and extended by
Higgins, J.P.T. and Thompson, S.G. "Controlling the risk of
spurious findings from meta-regression." *Stat. Med.*, 23(11):
1663-1682, 2004.

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
