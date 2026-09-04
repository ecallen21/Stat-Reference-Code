# Extreme Value Theory (Reference §38.1)

Coles (2001). Statistical modelling of rare events in the tails of a
distribution — floods, wind speeds, insurance losses, financial
crashes.

## Two workhorses

### Block maxima → GEV

For iid `X_i` and block size `n`, the standardised maximum
`(M_n − a_n) / b_n → GEV(μ, σ, ξ)`.

- `ξ < 0` — Weibull tail (bounded).
- `ξ = 0` — Gumbel tail.
- `ξ > 0` — Fréchet (heavy tail).

### Peaks over threshold (POT) → GPD

For a high threshold `u`, `(X − u) | X > u ~ GPD(σ_u, ξ)`. Same shape
parameter `ξ` as the GEV of the same underlying process.

## Return level & return period

The `T`-year return level under GEV is

```
z_T = μ + (σ/ξ) · ( (−log(1 − 1/T))^{−ξ} − 1 )
```

(and `μ − σ · log(−log(1 − 1/T))` in the Gumbel limit `ξ = 0`).

## When to use

- **Flood / storm / temperature** extremes.
- **Financial / insurance** VaR beyond historical maxima.
- **Reliability engineering** — worst-case component life.

## When NOT to use

- **Central-tendency questions** — EVT is designed for tails only.
- **Non-stationary processes** without covariates — refit a
  non-stationary GEV `μ(t)`, `σ(t)`.

## Files

- `python/extreme_value_theory.py` — GEV MLE + POT/GPD MLE +
  return-level computation. Demo: 40 years × 365 t₃ observations →
  GEV(μ=56.5, σ=14.1, ξ=0.24); 100-year return level ≈ 174 (block
  maxima), 223 (POT).
- `r/extreme_value_theory.R` — `extRemes::fevd`, `evd`, `texmex`
  (R); `scipy.stats.genextreme`/`genpareto`, `pyextremes` (Python).

## Assumptions & caveats

- **Threshold choice** in POT is critical — too low violates GPD
  approximation, too high wastes data. Use mean-residual-life or
  parameter-stability plots.
- **iid / stationarity** — dependent extremes need declustering (
  runs / intervals estimator).
- **Return-level extrapolation** — `T` beyond ~4× the record length
  is a leap of faith.
- **Small samples** — MLE for `ξ` is notoriously unstable with n < 25
  block maxima; consider L-moments or penalised likelihood.

## Related in this repo

- `tolerance-intervals` — parametric intervals for the same tails.
- `quantile-regression`, `expectile-regression` — conditional
  extremes.
- `mixture-models` — semiparametric tail-body models.

## Run

```
python techniques/extreme-value-theory/python/extreme_value_theory.py
Rscript techniques/extreme-value-theory/r/extreme_value_theory.R
```

**Refs:** Coles, S. *An Introduction to Statistical Modeling of Extreme Values*, Springer, 2001; Beirlant, J., Goegebeur, Y., Segers, J., & Teugels, J. *Statistics of Extremes*, Wiley, 2004.

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
