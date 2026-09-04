# Shewhart Control Charts (Reference §37.1)

Shewhart (1931). The **X-bar / R** (or S) chart tracks subgroup
**means and ranges** over time with 3-sigma control limits derived
from in-control subgroup statistics.

## Formulas

```
X-bar chart:  UCL / LCL  =  X̿  ±  A₂ · R̄
R chart:      UCL = D₄ · R̄,   LCL = D₃ · R̄
```

Constants `A₂, D₃, D₄` tabulated as functions of subgroup size `n`
(Montgomery Table VI).

## Western Electric run rules

- **Rule 1**: point > 3σ from centre.
- **Rule 2**: 2 of 3 consecutive > 2σ on the same side.
- **Rule 3**: 4 of 5 consecutive > 1σ on the same side.
- **Rule 4**: 8 consecutive on one side of centre.

## When to use

- **Manufacturing, healthcare, service** quality monitoring.
- **Continuous process** with subgrouped measurements.
- **First-pass diagnostic** before more advanced SPC.

## When NOT to use

- **Small shifts** — CUSUM / EWMA more sensitive.
- **Rare events / low count-per-subgroup** — G-chart / T-chart.
- **Multivariate quality** — Hotelling T² multivariate chart.

## Files

- `python/shewhart_control_charts.py` — from-scratch X-bar + R chart
  with Montgomery constants and all 4 Western-Electric rules. Demo:
  Phase-I baseline (subgroups 1-20) then 1.5-σ shift starting at
  subgroup 25. **Flags fire immediately at subgroup 26** (first
  post-shift) via rules 1-3.
- `r/shewhart_control_charts.R` — `qcc` (R reference); `pyspc`,
  `spc` (Python).

## Assumptions & caveats

- **Phase I vs Phase II** — establish limits from an in-control
  baseline, then monitor.
- **Subgroup design** — rational subgroups minimise within-subgroup
  variation; between-subgroup variation is the alarm signal.
- **Normality assumption** — R-chart robust; X-bar OK by CLT for
  n ≥ 4.
- **Run rules** increase false-alarm rate; use only for large-shift
  detection or with adjusted p-values.
- **Not for autocorrelated data** — model + apply to residuals.

## Related in this repo

- `cusum-charts`, `ewma-charts` — small-shift alternatives.
- `multivariate-control-charts` — Hotelling T² for correlated
  variables.
- `process-capability-indices`, `six-sigma-methods`,
  `acceptance-sampling`, `pareto-charts` — sibling SPC tools.
- `risk-adjusted-control-charts`, `rare-event-control-charts` —
  healthcare-specific variants.

## Run

```
python techniques/shewhart-control-charts/python/shewhart_control_charts.py
Rscript techniques/shewhart-control-charts/r/shewhart_control_charts.R
```

**Refs:** Shewhart, W. *Economic Control of Quality of Manufactured Product*, Van Nostrand, 1931; Montgomery, D.C. *Introduction to Statistical Quality Control*, 8th ed., Wiley, 2020.

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
