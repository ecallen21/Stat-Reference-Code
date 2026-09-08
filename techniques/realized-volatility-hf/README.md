# Realized Volatility from HF Data (Reference §47.75)

Andersen & Bollerslev (1998). Given intraday log-prices
`p_0, …, p_M`, realized variance:

    RV = Σᵢ (p_i − p_{i−1})²

is a consistent (as M → ∞) estimator of integrated variance for a
continuous semimartingale. **Bipower variation** (Barndorff-Nielsen–
Shephard 2004) is jump-robust:

    BV = (π/2) Σᵢ |r_i| · |r_{i−1}|

so `RV − BV → jump variation`. **Two-scales RV** (Zhang-Mykland-
Aït-Sahalia 2005) corrects for microstructure-noise bias.

## Files

- `python/realized_volatility_hf.py` — RV, BV, and two-scales
  RV from scratch. Demo (σ_true = 2 %/day):
  - `sqrt(RV)` → 2.00 % as sampling frequency M grows 12 → 1560
  - one 10-σ midday jump: RV=5.09e-4, BV=4.55e-4, jump variation
    5.4e-5 (BV isolates continuous part)
  - microstructure noise (multiplicative 0.1 %): RV inflated
    1.33e-3, two-scales estimator brings it back to 4.6e-4 near
    the clean 5.1e-4.
- `r/realized_volatility_hf.R` — `highfrequency`, `RTAQ`,
  `xts` (R); `arch`, custom (Python).

## When to use

- **Intraday equity / FX / crypto volatility** — trades or
  quote-midpoint sampled every few minutes.
- **HAR-RV forecasting** (Corsi 2009) — plug realized measures
  into ARFIMA-lite.
- **Volatility risk-premium** research — RV vs implied vol.
- **Jump detection** — BNS z-test on RV − BV.

## When NOT to use

- **Very sparse data** (once-a-day close) — RV degenerate; use
  GARCH / stochastic-vol models.
- **Extreme microstructure noise** — go to kernel-based realized
  variance (Barndorff-Nielsen et al 2008) or pre-averaging (Jacod).
- **Illiquid stocks** — sub-second returns dominated by bid-ask
  bounce; longer sampling frequency or noise-robust estimators.

## Assumptions & caveats

- **Semimartingale** — the underlying log-price is continuous
  (plus jumps) with finite quadratic variation.
- **No microstructure noise** — vanilla RV upward biased; two-
  scales / kernel RV / pre-averaging correct.
- **Sampling frequency vs noise** — trade-off (higher M → more
  bias); Zhang-Mykland-Aït-Sahalia gives asymptotics for optimal K.
- **Overnight returns** — not captured; add separately for
  daily-total variance.

## Related in this repo

- `garch`, `stochastic-volatility`, `regime-switching-markov` —
  parametric vol modelling.
- `arch`, `sarima-arimax`, `arfima`, `hierarchical-forecasting` —
  time-series alternatives.
- `euler-maruyama-sde`, `state-space-kalman`,
  `particle-filter-smc` — continuous-time / state-space cousins.
- `extreme-value-theory`, `cvar-expected-shortfall`, `copulas` —
  risk-management neighbours.

## Run

```
python techniques/realized-volatility-hf/python/realized_volatility_hf.py
Rscript techniques/realized-volatility-hf/r/realized_volatility_hf.R
```

**Refs:** Andersen, T.G. & Bollerslev, T. "Answering the skeptics: yes, standard volatility models do provide accurate forecasts." *Int Econ Rev* 39(4): 885-905, 1998; Barndorff-Nielsen, O.E. & Shephard, N. "Power and bipower variation with stochastic volatility and jumps." *J Financial Econom* 2(1): 1-37, 2004; Zhang, L., Mykland, P.A. & Aït-Sahalia, Y. "A tale of two time scales." *JASA* 100(472): 1394-1411, 2005.

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
