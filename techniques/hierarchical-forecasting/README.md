# Hierarchical Forecasting (Reference §13.37)

Multiple time series related by a summation hierarchy:

```
Total → A + B
A     → A1 + A2
B     → B1 + B2 + B3
```

Forecasts at every level must **aggregate coherently**. Independent per-level forecasts rarely do; **reconciliation** forces consistency and — with the right method — improves accuracy at every level.

## Summing matrix S

`S` is an `m × n_bottom` matrix encoding the aggregation. For the example above (`n_bottom = 5`), `S` has 8 rows: total, two groups, five bottom.

```
Reconciled  y_recon = S · G · ŷ_all
```

where `G` is an `m → n_bottom` mapping. Choices of `G` define the method.

## Bottom-up

`G = [0 | I_bottom]` — use only bottom-level forecasts, sum up.

- Coherent by construction.
- Ignores higher-level information; bottom series may be noisy.

## Top-down (Gross & Sohl)

`G = [p · e_1^T | 0]` — disaggregate the top forecast by historical proportions `p`.

- Denoises the top level.
- Doesn't respect bottom-level dynamics.

## MinT — Minimum Trace (Wickramasuriya, Athanasopoulos & Hyndman 2019)

```
G = (Sᵀ W⁻¹ S)⁻¹ Sᵀ W⁻¹
```

Uses `W`, the covariance of base-forecast errors at every level. Minimizes the trace of the reconciled-forecast error covariance under the coherence constraint — optimal under mild assumptions.

- **`W = I`** → OLS reconciliation.
- **`W = diag(σ²_i)`** → practical default (WLS reconciliation), needs only per-series error variances.
- **`W = shrinkage-covariance`** → full MinT, uses cross-series error correlation too.

## Files

- `python/hierarchical_forecasting.py` — builds `S` from a two-level dict, implements bottom-up, top-down, and diagonal-W MinT. Demo (5 bottom series, 3 aggregate + 1 total = 8 levels): incoherent MSE 43, bottom-up MSE 185, top-down MSE 227, MinT diag MSE 40 — MinT beats every other method and even improves on the base forecasts.
- `r/hierarchical_forecasting.R` — manual demo mirroring the Python; production: `hts::hts` + `hts::forecast(method = c("bu", "tdgsa", "comb"))`.

## When to use

- Retail (SKU → category → store → chain).
- Utilities (meter → substation → region → grid).
- Public health (hospital → county → state → country).
- Any forecasting problem where reported numbers must add up.

## Assumptions & caveats

- **Point-forecast focus** — MinT extends to prediction intervals (probabilistic reconciliation) via draw-and-reconcile Monte Carlo.
- **Base forecasts should be unbiased** at every level; MinT re-projects onto the coherent subspace but cannot fix bias.
- **Estimating W** — the shrinkage estimator (Schäfer-Strimmer) works well when the number of series is comparable to the training history.
- **Non-linear aggregations** (ratios, growth rates) don't have a matrix-`S` structure; use structural approaches instead.

## Run

```
python techniques/hierarchical-forecasting/python/hierarchical_forecasting.py
Rscript techniques/hierarchical-forecasting/r/hierarchical_forecasting.R
```

**Refs:** Hyndman, R.J., Ahmed, R.A., Athanasopoulos, G. & Shang, H.L. "Optimal combination forecasts for hierarchical time series." *Comput. Stat. Data Anal.* 55(9), 2579–2589, 2011; Wickramasuriya, S.L., Athanasopoulos, G. & Hyndman, R.J. "Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization." *JASA* 114(526), 804–819, 2019.

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
