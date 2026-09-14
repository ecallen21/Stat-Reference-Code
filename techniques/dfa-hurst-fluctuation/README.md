# Detrended Fluctuation Analysis (Reference §47.330)

Peng et al (1994). Estimate long-range correlations in
nonstationary time series:

1. Cumulative sum: `Y(k) = Σ_{i≤k}(x_i − mean)`
2. Partition into boxes of size `n`; detrend within each box.
3. `F(n) = √mean(residuals²)`
4. `F(n) ~ n^α`; `α` is the DFA exponent.

| α       | Interpretation                    |
|---------|-----------------------------------|
| 0.5     | uncorrelated (white noise)        |
| < 0.5   | anti-correlated                   |
| > 0.5   | long-range correlated (persistent)|
| 1.0     | 1/f noise                         |

## Files

- `python/dfa_hurst_fluctuation.py` — DFA-1 (linear detrend)
  on 4 known signals of length 4000. Recovers α ≈ 0.5
  (white), α ≈ 1.0-1.1 (1/f pink), α ≈ 1.5 (Brownian), α ≈
  0.0-0.1 (differenced noise) — matching theory.
- `r/dfa_hurst_fluctuation.R` — `nonlinearTseries::dfa`,
  `fractal::DFA`, `pracma::hurstexp` (R); `nolds.dfa`,
  `MFDFA`, `antropy.dfa`, from-scratch (Python).

## When to use

- **Long-memory detection** — finance returns, HRV, climate.
- **Non-stationary series** where windowed autocorrelation
  fails.
- **Preprocessing decision** — α > 1 flags need for
  differencing or fractional integration.

## When NOT to use

- **Very short series (< 500)** — box scaling is unstable.
- **Signals with strong periodicity** — DFA can misestimate;
  use MFDFA or DFA-2 (polynomial detrend).
- **When precise scaling exponent is needed** — MFDFA and
  wavelet-leader analysis are more accurate.

## Assumptions & caveats

- **Detrend order** — DFA-1 = linear; DFA-2 = quadratic
  (handles cubic trends).
- **Scale range** — usually `8` to `N/4`; crossover between
  scales is diagnostic.
- **Confidence intervals** — bootstrap over boxes;
  asymptotic theory exists but assumes ergodicity.
- **Multifractal extension (MFDFA)** — reveals scaling
  variation across signal amplitudes.

## Related in this repo

- `sample-entropy-signal` — companion complexity measure.
- `hilbert-transform-analytic` — nonstationary features.
- `wavelet-analysis` — related multiresolution tool.
- `arfima` — fractional-integration model related to α.

## Run

```
python techniques/dfa-hurst-fluctuation/python/dfa_hurst_fluctuation.py
Rscript techniques/dfa-hurst-fluctuation/r/dfa_hurst_fluctuation.R
```

**Refs:** Peng, C.-K., Buldyrev, S.V., Havlin, S., Simons, M., Stanley, H.E. and Goldberger, A.L. "Mosaic organization of DNA nucleotides." *Phys. Rev. E*, 49(2): 1685-1689, 1994.

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
