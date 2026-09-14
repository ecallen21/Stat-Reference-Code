# Cumulative Meta-Analysis (Reference §47.274)

Lau, Antman, Chalmers et al (1992). Repeat the pooled
meta-analysis after each new study is added (usually
chronologically), showing HOW THE EVIDENCE ACCUMULATED. Reveals:

- when the pooled effect first crossed significance
- whether later studies reinforced or contradicted early ones
- whether the current CI has converged

## Files

- `python/cumulative_meta_analysis.py` — DL-based cumulative
  pooling ordered by year. Demo: 12 studies from 1980-2000
  with true effect µ=0.4. Table shows pooled estimate, SE,
  and 95% CI at each cumulative K. First cumulative CI to
  exclude zero flagged as "first crossed sig".
- `r/cumulative_meta_analysis.R` — `metafor::cumul`,
  `meta::metacum`, `dmetar::EvidenceCumulPlot` (R);
  PythonMeta, from-scratch (Python).

## When to use

- **Evidence-based-medicine audits** — Lau et al's original
  motivation was showing that streptokinase evidence was
  conclusive years before textbooks caught up.
- **Deciding whether more studies are needed** — a converged
  cumulative CI signals "further studies unlikely to change
  the conclusion".
- **Communicating with stakeholders** — the visual timeline
  is intuitive.

## When NOT to use

- **Very few studies** — cumulative dynamics are noisy at
  K < 6.
- **Highly heterogeneous studies** — the pooled trajectory
  bounces around; combine with meta-regression for stability.
- **When ordering is ambiguous** — no natural chronological /
  quality order → cumulative plot is arbitrary.

## Assumptions & caveats

- **Ordering choice** — chronological is most common; sample
  size or quality gives alternative narratives.
- **Interpretation caveat** — sequential re-testing does NOT
  inflate α at the meta-analysis level (unlike interim
  looks in a single trial) because the studies are already
  fixed; the cumulative CI is a description, not a decision
  rule.
- **Combine with prospective monitoring** — Wetterslev et al's
  Trial Sequential Analysis extends cumulative meta with
  O'Brien-Fleming-style boundaries.
- **Report bare estimates AND CI** — the estimate can drift
  before the CI narrows.

## Related in this repo

- `dersimonian-laird-random-effects` — the underlying pool.
- `leave-one-out-meta` — the "remove one at a time" cousin.
- `sequential-analysis` — TSA connection.

## Run

```
python techniques/cumulative-meta-analysis/python/cumulative_meta_analysis.py
Rscript techniques/cumulative-meta-analysis/r/cumulative_meta_analysis.R
```

**Refs:** Lau, J., Antman, E.M., Jimenez-Silva, J., Kupelnick, B., Mosteller, F. and Chalmers, T.C. "Cumulative meta-analysis of therapeutic trials for myocardial infarction." *N. Engl. J. Med.*, 327(4): 248-254, 1992.

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
