# Rare-Event Control Charts (Reference §37.11)

For low-defect-rate processes the Shewhart p / np chart is
uninformative (most subgroups show 0 events). Alternatives:

## G-chart (geometric)

Plot **number of non-events between consecutive events** — G ~ Geom(p).
Centre `1/p`, `σ = √((1−p)/p²)`, limits `μ ± 3σ`.

## T-chart (exponential)

Plot **time between events** — under a Poisson process
`T ~ Exp(λ)`, centre `1/λ`.

## Bernoulli CUSUM (Reynolds-Stoumbos 1999)

Log-likelihood-ratio CUSUM for detecting a shift `p_0 → p_1`.

## When to use

- **Hospital-acquired infections, adverse events, surgical mortality**.
- **Manufacturing near-zero-defect processes** (aerospace, pharma).
- **Any low-rate binary or count process**.

## When NOT to use

- **Common-rate processes** — Shewhart p / np-chart works.
- **Sub-groupable data with subgroup-level variability** — use
  standard p-charts.

## Files

- `python/rare_event_control_charts.py` — G-chart limits + Bernoulli
  CUSUM. Demo baseline `p_0=0.005 → shifted p_1=0.02` at t=1500.
  **Bernoulli CUSUM (h=5) detects at t=1555 (delay 55)**; defects
  clearly increase (11 → 31 in equal-size windows).
- `r/rare_event_control_charts.R` — `spc` (R reference), `qcc`
  adjacent; custom (Python).

## Assumptions & caveats

- **Small counts** — G-chart limits based on geometric distribution
  are asymmetric; standard 3-σ limits may cross 0.
- **Independent events** — G assumes iid; correlated events (patient
  clusters) inflate variance.
- **Rate estimation** — baseline `p_0` from a Phase I window;
  contaminated Phase I biases limits.
- **Bernoulli CUSUM** more powerful for detecting *sustained* shifts;
  G-chart better for step-like changes in *individual* intervals.

## Related in this repo

- `cusum-charts`, `ewma-charts`, `shewhart-control-charts` — general
  SPC.
- `risk-adjusted-control-charts` — healthcare-specific with risk
  adjustment.
- `poisson-regression` (if present) — modelling rare-event rates.

## Run

```
python techniques/rare-event-control-charts/python/rare_event_control_charts.py
Rscript techniques/rare-event-control-charts/r/rare_event_control_charts.R
```

**Refs:** Benneyan, J.C. "Statistical quality control methods in infection control and hospital epidemiology." *Infection Control and Hospital Epidemiology*, 1998; Reynolds, M.R. & Stoumbos, Z.G. "A CUSUM chart for monitoring a proportion when inspecting continuously." *Journal of Quality Technology*, 1999.

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
