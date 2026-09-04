# CUSUM Control Chart (Reference §37.2)

Page (1954). **Cumulative-sum** chart for detecting SMALL PERSISTENT
SHIFTS faster than Shewhart.

## Tabular CUSUM

```
S_i^+ = max(0, S_{i−1}^+ + (x_i − μ_0 − k · σ))
S_i^- = min(0, S_{i−1}^- + (x_i − μ_0 + k · σ))
```

Signal when `S_i^+ > h · σ` or `S_i^- < −h · σ`.

## Standard tuning

- `k = δ / 2` where `δ` is the shift size (in σ units) you want to
  detect fastest.
- `h = 4` or `5` — controls ARL_0 (in-control average run length).

Design tables give `(h, k)` for target `(ARL_0, ARL_1)`.

## When to use

- **Small-to-moderate shifts** (0.5σ – 1.5σ) where Shewhart is slow.
- **Continuous monitoring** where every observation matters.
- **Change-point detection** in time series.

## When NOT to use

- **Very large shifts** — Shewhart detects them in one point anyway.
- **Autocorrelated series** — model + apply CUSUM to residuals.

## Files

- `python/cusum_charts.py` — from-scratch tabular CUSUM +
  Shewhart comparison. Demo: 1-σ shift at t=100, n=200:
  **CUSUM (k=0.5, h=6) signals at t=116 (delay 16)**; Shewhart
  (3-σ) never signals in the 100 post-shift samples.
- `r/cusum_charts.R` — `qcc::cusum`, `spc`, `rQCC` (R); `pyspc`
  (Python).

## Assumptions & caveats

- **h tuning** — `h = 4` gives ARL_0 ≈ 168 for k=0.5;
  `h = 5` → ARL_0 ≈ 465; `h = 6` → ARL_0 ≈ 1200.
- **False-alarm control** — smaller `h` = faster detection but more
  false alarms.
- **Restart after signal** — reset `S^+` = `S^-` = 0.
- **Self-starting CUSUM** (Hawkins-Olwell) if μ_0 unknown.
- **Bernoulli CUSUM** for binary rare events (see
  `rare-event-control-charts`).

## Related in this repo

- `shewhart-control-charts` — large-shift chart.
- `ewma-charts` — small-shift alternative.
- `concept-drift-adwin` — ML-focused drift detection sibling.
- `rare-event-control-charts` — Bernoulli CUSUM variant.
- `sequential-analysis` — SPRT is the Bayesian-flavour sibling.

## Run

```
python techniques/cusum-charts/python/cusum_charts.py
Rscript techniques/cusum-charts/r/cusum_charts.R
```

**Refs:** Page, E.S. "Continuous inspection schemes." *Biometrika*, 1954; Hawkins, D.M. & Olwell, D.H. *Cumulative Sum Charts and Charting for Quality Improvement*, Springer, 1998.

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
