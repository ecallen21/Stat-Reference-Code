# Croston / SBC — Intermittent Demand (Reference §47.350)

Croston (1972); Syntetos-Boylan correction (2005). Forecast
sporadic demand series (many zeros) by SEPARATELY smoothing
the non-zero demand sizes and the inter-arrival times:

```
z_hat = SES(non-zero demand sizes, α)
x_hat = SES(inter-arrival times, α)
forecast rate = z_hat / x_hat
```

Syntetos-Boylan-Croston (SBC) multiplies the rate by
`(1 − α/2)` to remove Croston's inbuilt bias, making it the
default in modern software.

## Files

- `python/croston_intermittent_demand.py` — synthetic
  intermittent series (n=60, event probability 0.3, Poisson-
  10 sizes) with true rate 3.0 (empirical mean 2.23 due to
  sample variability). Croston(α=0.1) = 1.83, SBC = 1.74;
  α=0.3 pulls tighter to recent runs. Illustrates the
  smoothing-vs-bias trade-off.
- `r/croston_intermittent_demand.R` — `forecast::croston`,
  `tsintermittent::crost`/`tsb` (R);
  `statsforecast.IMAPA`, `sktime.CrostonForecaster`,
  from-scratch (Python).

## When to use

- **Spare-parts, medical supplies, luxury retail** — series
  with long stretches of zeros.
- **Inventory-management forecasting** — service-level
  planning needs a rate estimate.
- **Baseline** for evaluating newer intermittent-demand
  models (TSB, IMAPA, ADIDA).

## When NOT to use

- **Regular (non-intermittent) demand** — SES / Holt-Winters
  is simpler and better.
- **When demand structure changes over time** — Croston
  assumes stationary rate.
- **When zero-inflation model / hurdle is appropriate** —
  those give proper probabilistic forecasts.

## Assumptions & caveats

- **Bias** — original Croston is upward-biased; use SBC or
  TSB (Teunter-Syntetos-Babai) instead.
- **α selection** — 0.05-0.3 common; MSE / MASE minimisation
  over held-out.
- **Zero forecasts** — Croston never forecasts 0; if the
  series may go permanently to zero, use hurdle models.
- **Distinction from ADIDA** — ADIDA aggregates then
  disaggregates; Croston works at the raw resolution.

## Related in this repo

- `exponential-smoothing`, `holt-winters-forecasting`,
  `theta-method-forecast` — regular-demand cousins.
- `zero-inflated-regression`, `hurdle-model` —
  cross-sectional intermittent alternatives.
- `poisson-regression`, `negative-binomial-regression` —
  count regressions for zero-heavy data.
- `count-time-series` — dynamic count models.

## Run

```
python techniques/croston-intermittent-demand/python/croston_intermittent_demand.py
Rscript techniques/croston-intermittent-demand/r/croston_intermittent_demand.R
```

**Refs:** Croston, J.D. "Forecasting and stock control for intermittent demands." *Oper. Res. Q.*, 23(3): 289-303, 1972; Syntetos, A.A., Boylan, J.E. and Croston, J.D. "On the categorization of demand patterns." *J. Oper. Res. Soc.*, 56: 495-503, 2005.

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
