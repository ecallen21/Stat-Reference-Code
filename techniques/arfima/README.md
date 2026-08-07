# ARFIMA — Fractionally Integrated ARIMA (Reference §13.16)

Standard ARIMA(p, d, q) allows integer `d`. **ARFIMA generalizes to fractional `d ∈ (−0.5, 0.5)`**, capturing **long memory** — slowly-decaying autocorrelations of the form `ρ(k) ~ k^(2d − 1)` rather than the exponential decay of an ARMA process.

- `d ∈ (0, 0.5)` — long memory (positive slowly-decaying autocorrelations).
- `d = 0` — ordinary short-memory ARMA.
- `d ∈ (−0.5, 0)` — anti-persistent (negative long-range dependence).

## Fractional differencing (Granger-Joyeux-Hosking 1980)

```
(1 − L)^d = Σ_{k≥0}  π_k L^k       π_k = (−1)^k Γ(d+1) / (Γ(k+1) Γ(d−k+1))
```

Recursion: `π_0 = 1`, `π_k = π_{k−1} · (k − 1 − d) / k`. In practice truncate the sum at `k = O(√T)` – tail weights are `O(k^(−1−d))`.

## Estimation of `d`

- **Whittle spectral MLE** — frequency-domain likelihood; efficient, low variance. `fracdiff::fracdiff` in R.
- **Geweke-Porter-Hudak (GPH) log-periodogram regression** —

```
log I(ω_j) = c − 2 d log|2 sin(ω_j / 2)| + ε_j            for j = 1, ..., m
```

Regress the low-frequency log-periodogram on the log-frequency covariate. Bandwidth `m = T^α` with `α ∈ (0.5, 0.8)` balances bias vs variance; asymptotic `SE(d̂) = √(π² / 24m)`.

## Files

- `python/arfima.py` — from-scratch fractional-differencing weights, ARFIMA(0, d, 0) simulation, and GPH estimator. Demo on T = 2000: recovers d = 0.17 (true 0.20), d = 0.31 (true 0.35), d = 0.00 (true 0.0) with `m = T^0.7 ≈ 204`, all within one asymptotic SE.
- `r/arfima.R` — `fracdiff::fracdiff` Whittle MLE; more efficient than GPH.

## When to use

- Financial / hydrological / geophysical time series with visibly slow ACF decay.
- Long memory in volatility (see `stochastic-volatility` and FIGARCH).
- Series where the differenced version (I(1)) looks over-differenced and the level version looks non-stationary — ARFIMA is the middle ground.

## Assumptions

- Second-order stationarity requires `d < 0.5`; for `d ∈ [0.5, 1)` the series is non-stationary but mean-reverting.
- Short-memory ARMA structure can be layered on top: after fractional differencing, fit an ARMA(p, q) to the residuals.
- GPH assumes the low-frequency spectral density behaves like `f(ω) ~ ω^(−2d)`.

## Run

```
python techniques/arfima/python/arfima.py
Rscript techniques/arfima/r/arfima.R
```

**Refs:** Granger, C.W.J. & Joyeux, R. "An introduction to long-memory time series models and fractional differencing." *J. Time Ser. Anal.* 1(1), 15–29, 1980; Hosking, J.R.M. "Fractional differencing." *Biometrika* 68(1), 165–176, 1981; Geweke, J. & Porter-Hudak, S. "The estimation and application of long memory time series models." *J. Time Ser. Anal.* 4(4), 221–238, 1983.

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
