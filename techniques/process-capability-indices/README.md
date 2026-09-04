# Process Capability Indices (Reference §37.7 / §37.12)

Summaries of how well a process fits within specification limits.

## Formulas

```
Cp   =  (USL − LSL) / (6 σ)                   potential capability (centred)
Cpk  =  min( (USL − μ)/(3σ), (μ − LSL)/(3σ) ) centring-aware
Pp   =  (USL − LSL) / (6 S)                   long-term (overall S)
Ppk  =  min( (USL − μ)/(3S), (μ − LSL)/(3S) )
Cpm  =  (USL − LSL) / (6 √(σ² + (μ − T)²))    Taguchi target-based
```

## Interpretation

- `Cp ≥ 1.33` — commonly considered **capable**.
- `Cp = Cpk` when the process is exactly centred.
- `Cpk < Cp` reveals **off-centre** processes.
- `Cpm` penalises drift from a target `T` even when within spec.

## When to use

- **Process qualification / audit** — regulatory Six Sigma reports.
- **Supplier scoring** — Cpk thresholds in contracts.
- **Root-cause analysis** — compare Cp vs Cpk to spot centring vs
  variability issues.

## When NOT to use

- **Non-normal / heavy-tailed data** — use non-normal capability
  indices (Clements 1989).
- **Autocorrelated data** — inflated Cp; model + apply.
- **Small n** — indices have wide CI; report bounds.

## Files

- `python/process_capability_indices.py` — from-scratch Cp / Cpk /
  Pp / Ppk / Cpm. Demo:
  - **Case A** (centred, σ=0.6): Cp = Cpk = 1.08 (all indices agree).
  - **Case B** (off-centre μ=11 with same σ): Cp = 1.14 but Cpk =
    0.57; Cpm = 0.58 — off-centre process caught by Cpk / Cpm.
- `r/process_capability_indices.R` — `qcc::process.capability`,
  `SixSigma` (R); `pyspc` (Python).

## Assumptions & caveats

- **Short-term vs long-term SD** — Cp/Cpk use short-term (within-
  subgroup); Pp/Ppk use overall. Diff quantifies process shift over
  time.
- **Normality** — indices are Gaussian-centric; check Q-Q or use
  Bothe (1997) tolerance-interval alternatives.
- **CIs** — bootstrap or normal-approximation (Kirmani-Kocherlakota).
- **One-sided specs** — use CPU or CPL alone.
- **Non-parametric alternative**: (P95 − P5) / (some spread) —
  distribution-free capability.

## Related in this repo

- `shewhart-control-charts`, `cusum-charts`, `ewma-charts` —
  monitoring cousins (stability).
- `six-sigma-methods` — DPMO ↔ sigma level ↔ Cp/Cpk mapping.
- `acceptance-sampling` — decision after capability assessment.
- `multi-vari-charts`, `pareto-charts` — root-cause tools.

## Run

```
python techniques/process-capability-indices/python/process_capability_indices.py
Rscript techniques/process-capability-indices/r/process_capability_indices.R
```

**Refs:** Kane, V.E. "Process capability indices." *Journal of Quality Technology*, 1986; Taguchi, G. *Introduction to Quality Engineering*, Asian Productivity Organization, 1986; Montgomery, D.C. *Introduction to Statistical Quality Control*, 8th ed., Wiley, 2020.

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
