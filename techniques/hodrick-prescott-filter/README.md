# Hodrick-Prescott Filter (Reference §47.143)

Hodrick & Prescott (1997). Decomposes a time series y_t into a
smooth trend τ_t and cycle c_t = y_t − τ_t by minimising:

    sum_t (y_t − τ_t)² + λ sum_t ((τ_{t+1} − τ_t) − (τ_t − τ_{t−1}))².

Standard λ conventions: 100 (annual), 1600 (quarterly), 129600
(monthly). Closed-form linear solve: τ = (I + λ K'K)⁻¹ y, where K
is the 2nd-difference operator matrix.

## Files

- `python/hodrick_prescott_filter.py` — sparse-linear-algebra HP
  filter (scipy.sparse) on 200-point series (quadratic trend +
  20-period sinusoid + N(0, 0.5²)):
  - λ = 10: trend RMSE = 1.33, cycle RMSE = 1.35 (trend hugs data).
  - **λ = 1600** (quarterly default): trend RMSE = **0.30**, cycle
    RMSE = **0.54** — trend captures the quadratic curve cleanly.
  - λ = 129600: trend RMSE = 0.18 (near-linear), cycle absorbs
    curvature + oscillation.
- `r/hodrick_prescott_filter.R` — `mFilter::hpfilter`; siblings
  `bkfilter` (Baxter-King bandpass) and `cffilter` (Christiano-
  Fitzgerald).

## When to use

- **Macroeconomic trend-cycle decomposition** (GDP, employment,
  inflation).
- **Detrending** for subsequent time-series analysis (variance
  decomposition, business-cycle correlations).
- **Smoother** with a single tunable knob.

## When NOT to use

- **Structural-break** series — HP smooths across breaks; use
  segmented or Bai-Perron models.
- **Real-time / end-of-sample** work — HP has known end-point bias;
  Hamilton (2018) proposes a regression-based alternative.
- **Highly seasonal** series without prior deseasonalisation.

## Assumptions & caveats

- **λ choice** is subjective; Ravn-Uhlig (2002) proposes scaling
  λ ∝ (freq)⁴.
- **End-point bias** — the trend near the boundary shifts a lot
  when new data arrive.
- **Spurious cycles** — HP can generate cycles from I(1) or I(2)
  series (Hamilton 2018 critique).
- **Sparse solve is O(T)** — linear in series length.

## Related in this repo

- `savitzky-golay-filter`, `butterworth-bandpass` — alternative
  low-pass smoothers.
- `state-space-models`, `structural-time-series` — model-based
  decompositions with SEs.
- `panel-cointegration`, `arima-modeling` — different flavours of
  time-series analysis.

## Run

```
python techniques/hodrick-prescott-filter/python/hodrick_prescott_filter.py
Rscript techniques/hodrick-prescott-filter/r/hodrick_prescott_filter.R
```

**Refs:** Hodrick, R. J. & Prescott, E. C. "Postwar U.S. business cycles: an empirical investigation." *JMCB* 29(1), 1997; Ravn, M. O. & Uhlig, H. "On adjusting the Hodrick-Prescott filter for the frequency of observations." *Rev Econ Stat* 84(2), 2002; Hamilton, J. D. "Why you should never use the Hodrick-Prescott filter." *Rev Econ Stat* 100(5), 2018.

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
