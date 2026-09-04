# EWMA Control Chart (Reference §37.3)

Roberts (1959). **Exponentially-weighted moving-average** chart:

```
z_i = λ x_i + (1 − λ) z_{i−1},   z_0 = μ_0.
```

Variance grows to a steady state:

```
σ_z²  =  σ²  ·  (λ / (2 − λ))  ·  (1 − (1 − λ)^{2i}).
```

Limits at `μ_0 ± L · σ_z`.

## Tuning

- **λ = 0.05 – 0.20** for small shifts (< 1σ).
- **λ = 0.20 – 0.40** for medium shifts.
- **λ → 1** recovers Shewhart.
- `L = 2.7 – 3.5` chosen jointly with λ for the target ARL.

## When to use

- **Small shift detection** without the tabular CUSUM interpretation
  overhead.
- **Autocorrelated series** — EWMA of residuals from an AR model.
- **Continuous processes** where every point should update the chart.

## When NOT to use

- **Very large shifts** — Shewhart suffices.
- **Very small n** — steady-state variance not reached.

## Files

- `python/ewma_charts.py` — from-scratch EWMA with time-varying
  variance limits. Demo: 0.75-σ shift at t=100, n=200:
  **λ = 0.1, 0.2 detect at t=119 (delay 19)**; λ = 0.4 slower at
  t=137 (delay 37). Confirms smaller λ = better for smaller shifts.
- `r/ewma_charts.R` — `qcc::ewma`, `spc` (R); `pyspc` (Python).

## Assumptions & caveats

- **λ and L jointly designed** — Montgomery tables give (λ, L) for
  target ARL_0 = 370 and various shift sizes.
- **Steady-state vs time-varying limits** — use time-varying to
  avoid inflated early false-alarm rate.
- **Restart after signal** — reset `z` to `μ_0`.
- **Comparison to CUSUM** — EWMA and CUSUM have similar ARL curves;
  EWMA is often preferred for plotting continuity.

## Related in this repo

- `cusum-charts` — the algorithmic cousin.
- `shewhart-control-charts` — large-shift parent.
- `model-monitoring-metrics` — MLOps analogue with EWMA baselines.
- `exponential-smoothing` (if present) — same smoother in forecasting.

## Run

```
python techniques/ewma-charts/python/ewma_charts.py
Rscript techniques/ewma-charts/r/ewma_charts.R
```

**Refs:** Roberts, S.W. "Control chart tests based on geometric moving averages." *Technometrics*, 1959; Montgomery, D.C. *Introduction to Statistical Quality Control*, 8th ed., Wiley, 2020 (Ch. 9).

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
