# Stochastic Volatility (Reference §13.40)

Latent-state alternative to GARCH. Log-variance follows its own **stochastic** AR(1) process rather than a deterministic function of past `y` and past variance:

```
y_t  = exp(h_t / 2) · ε_t          ε_t ~ N(0, 1)
h_t  = μ + φ (h_{t−1} − μ) + σ_η · η_t     η_t ~ N(0, 1)
```

- `μ` — mean of log-volatility
- `φ ∈ (−1, 1)` — persistence (usually near 1 in equity data)
- `σ_η` — log-volatility shock scale

## SV vs GARCH

|                | GARCH(1,1)                        | SV                                    |
|----------------|-----------------------------------|---------------------------------------|
| Volatility     | deterministic given `y_{<t}`      | stochastic latent process             |
| Estimation     | closed-form conditional MLE       | Kalman quasi-MLE / MCMC / particle    |
| Empirical fit  | good; well-calibrated tails       | often slightly better on equity data  |
| Extensions     | GJR, EGARCH, IGARCH               | jumps, leverage, multivariate SV      |

## Estimation

- **Quasi-MLE (Harvey-Ruiz-Shephard)** — Kalman filter on `log y_t² = h_t + log ε_t²`, treating `log ε_t²` as Normal (approximation). Fast; slight bias in `σ_η`.
- **Full Bayesian MCMC (Kim-Shephard-Chib 1998)** — mixture-of-normals approximation to `log χ²_1` with Gibbs updates on the latent path. Gold standard. `stochvol` in R, custom Stan / PyMC models in Python.
- **Particle filter** — exact likelihood via sequential Monte Carlo; used for parameter estimation (particle MCMC) or model comparison.

## Files

- `python/stochastic_volatility.py` — simulator, bootstrap particle filter for the latent `h_t` given parameters, and Kalman-filter QMLE for `(μ, φ, σ_η)`. Demo on T = 500 recovers `μ = −8.24` (true −8), `φ = 0.977` (true 0.97), `σ_η = 0.14` (true 0.15) — all within one SD.
- `r/stochastic_volatility.R` — `stochvol::svsample` for the standard Kim-Shephard-Chib MCMC.

## When to use

- **Financial returns** — equity, FX, commodity — with visible volatility clustering and slight lead-lag between returns and volatility.
- **Latent-state modelling** where you want a probabilistic decomposition of observed variability into a signal component and an evolving noise scale.
- As the volatility layer inside a fuller model — SV + jumps, SV-AR mean, multivariate SV.

## Assumptions & caveats

- **Identifiability** — `φ` and `σ_η` trade off in short samples; T ≥ 500 gives reasonable QMLE recovery, T ≥ 2000 for tight intervals.
- **Non-stationarity** at `φ → 1` — imposes prior structure (`phi ~ Beta(20, 1.5)` in `stochvol` for stable estimation).
- **`log ε² ~ log χ²_1`** — non-Normal; Kalman-QMLE is efficient only asymptotically.

## Run

```
python techniques/stochastic-volatility/python/stochastic_volatility.py
Rscript techniques/stochastic-volatility/r/stochastic_volatility.R
```

**Refs:** Taylor, S.J. "Modelling stochastic volatility." *Math. Finance* 4(2), 183–204, 1994; Harvey, A., Ruiz, E. & Shephard, N. "Multivariate stochastic variance models." *Rev. Econ. Stud.* 61(2), 247–264, 1994; Kim, S., Shephard, N. & Chib, S. "Stochastic volatility: likelihood inference and comparison with ARCH models." *Rev. Econ. Stud.* 65(3), 361–393, 1998.

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
