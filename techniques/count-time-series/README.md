# Count Time Series (Reference §13.32)

Discrete non-negative-integer time series (accident counts, event counts, monthly hospitalizations) need count-specific models — Gaussian ARMA violates non-negativity and integer support.

## INAR(1) — Integer AR via binomial thinning

McKenzie 1985; Al-Osh & Alzaid 1987.

```
y_t = α ∘ y_{t−1} + ε_t
```

where `α ∘ y` is **binomial thinning**: given `y_{t−1}`, draw `Binomial(y_{t−1}, α)` independently. `ε_t` is an integer innovation (Poisson(λ) is standard).

**Marginal**: for Poisson innovations the stationary distribution is `Poisson(λ / (1 − α))`.

**MoM estimator**: `α̂ = ρ̂(1)`, `λ̂ = ȳ (1 − α̂)`.

## INGARCH(1, 1) — Poisson observation-driven

Ferland-Latour-Oraichi 2006 (also called Poisson-AR).

```
y_t | past  ~ Poisson(μ_t)
μ_t         = ω + α y_{t−1} + β μ_{t−1}
```

Deterministic conditional-mean recursion; parallels GARCH for counts. Fit by conditional Poisson MLE.

## Files

- `python/count_time_series.py` — simulator + method-of-moments estimator for Poisson-INAR(1), and simulator + conditional Poisson MLE for INGARCH(1, 1). Demo (n = 1000): INAR recovers `α = 0.587` (true 0.6), `λ = 2.54` (true 2.5); INGARCH recovers `ω = 0.95` (true 1.0), `α = 0.48` (true 0.4), `β = 0.24` (true 0.3).
- `r/count_time_series.R` — `tscount::tsglm(family = "poisson", model = list(past_obs = 1, past_mean = 1))` — the reference INGARCH implementation.

## When to use

- Low-count series where Gaussian approximation is poor (`ȳ < 5`).
- Series where negative or non-integer forecasts would violate downstream constraints (e.g. discrete resource planning).
- Volatility / clustering in the intensity itself — INGARCH captures self-exciting bursts of activity.

## Related methods

- **Zero-inflated** and **hurdle** extensions for excess zeros.
- **Negative-Binomial** observation family for over-dispersion.
- **State-space Poisson** with a Gaussian latent (INAR-style but with continuous state) — Ord, Fernandes & Harvey 1993.
- **Hawkes processes** for self-exciting event times (as opposed to counts on a fixed grid).

## Assumptions & caveats

- **Stationarity**: INGARCH needs `α + β < 1`; INAR needs `|α| < 1`.
- **MoM efficiency**: MoM is a fine starting point for INAR; ML is available in `tscount` for both.
- **Over-dispersion**: real counts often show `Var > Mean`; use Negative-Binomial INGARCH (`tscount(distr = "nbinom")`).

## Run

```
python techniques/count-time-series/python/count_time_series.py
Rscript techniques/count-time-series/r/count_time_series.R
```

**Refs:** McKenzie, E. "Some simple models for discrete variate time series." *Water Resour. Bull.* 21(4), 645–650, 1985; Al-Osh, M.A. & Alzaid, A.A. "First-order integer-valued autoregressive (INAR(1)) process." *J. Time Ser. Anal.* 8(3), 261–275, 1987; Ferland, R., Latour, A. & Oraichi, D. "Integer-valued GARCH process." *J. Time Ser. Anal.* 27(6), 923–942, 2006.

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
