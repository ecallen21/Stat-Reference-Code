# Six Sigma Statistical Methods (Reference §37.9)

Motorola (1986). Quality framework combining **DMAIC** (Define /
Measure / Analyse / Improve / Control) with characteristic
calculations.

## Signature calculations

- **DPMO** = `defects / (units · opportunities/unit) · 10⁶`.
- **Sigma level** = `Φ⁻¹(1 − DPMO/10⁶) + 1.5` (Motorola convention).
- **Yield** = `1 − DPMO/10⁶`.

## Standard mapping (with 1.5-σ shift)

| σ | DPMO | Yield |
|---|-----|------|
| 3 | 66 807 | 93.32 % |
| 4 | 6 210 | 99.379 % |
| 5 | 233 | 99.977 % |
| 6 | 3.4 | 99.99966 % |
| 7 | 0.019 | 99.9999981 % |

## When to use

- **Formal quality-improvement programmes**.
- **Executive-level reporting** — sigma level is a universal shorthand.
- **Supplier scorecards** — DPMO in contracts.

## When NOT to use

- **The 1.5-σ shift** is a Motorola convention, not derived; some
  statisticians criticise it as arbitrary.
- **Non-normal quality characteristics** — DPMO from tail probabilities
  needs care.

## Files

- `python/six_sigma_methods.py` — DPMO ↔ sigma ↔ yield conversions
  + DMAIC checklist. Standard sigma table (3 → 66 807 DPMO through
  6 → 3.4). Empirical demo: 42 defects in 3 500 units × 3
  opportunities → DPMO 4 000, sigma ≈ 4.15, yield 99.60 %.
- `r/six_sigma_methods.R` — `SixSigma` (R); `sixsigma` (Python).

## Assumptions & caveats

- **1.5-σ shift is a convention** — some organisations omit it
  ("short-term sigma").
- **DPMO requires clearly defined "opportunities"**; inflating
  opportunities gives fake sigma.
- **DMAIC** is a methodology, not a formula — the pipeline shown is a
  checklist to prompt the right tools.
- **Six Sigma vs Lean Six Sigma** — Lean adds waste-reduction tools
  (Value Stream Mapping).

## Related in this repo

- `process-capability-indices` — Cp/Cpk relate directly to sigma
  levels.
- `shewhart-control-charts`, `cusum-charts`, `ewma-charts` — the
  "Control" phase tools.
- `pareto-charts`, `multi-vari-charts` — the "Analyse" phase tools.
- `design-of-experiments` (if present) — the "Improve" phase engine.

## Run

```
python techniques/six-sigma-methods/python/six_sigma_methods.py
Rscript techniques/six-sigma-methods/r/six_sigma_methods.R
```

**Refs:** Motorola Inc. "Six Sigma quality initiative." 1986; Pyzdek, T. & Keller, P. *The Six Sigma Handbook*, 5th ed., McGraw-Hill, 2018.

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
