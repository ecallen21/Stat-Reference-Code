# Empirical Mode Decomposition (EMD) (Reference §47.93)

Huang et al (1998). Data-driven decomposition of a signal into
Intrinsic Mode Functions (IMFs) via SIFTING:

    1. Interpolate max / min envelopes with cubic splines.
    2. m(t) = mean of envelopes.
    3. h(t) = x(t) − m(t).
    4. Repeat until h is an IMF (equal # zero-crossings and
       extrema, zero mean envelope).

Residue r = x − IMF; iterate for next IMF. Unlike Fourier / wavelet,
EMD makes no basis assumption; the Hilbert-Huang transform combines
EMD + instantaneous frequency for nonlinear / non-stationary
analysis.

## Files

- `python/empirical_mode_decomposition.py` — from-scratch EMD
  with mirror-extended cubic-spline envelopes (numerical stability).
  Demo (N=800 signal = 1 Hz + 3 Hz amp-modulated + linear trend +
  noise):
  - IMF 2 dominant freq ≈ 4.8 Hz (captures 3 Hz)
  - IMF 3 dominant freq ≈ 1.6 Hz (captures 1 Hz)
  - Residue tracks the linear trend (mean 1.47).
- `r/empirical_mode_decomposition.R` — `Rlibeemd`, `EMD`,
  `hht` (R); `PyEMD`, `EMD-signal`, from-scratch (Python).

## When to use

- **Nonlinear / non-stationary time series** — biomedical (ECG,
  EEG), geophysical (climate), turbulence.
- **Adaptive detrending** — the residue is a data-driven trend.
- **Hilbert-Huang transform** — pair with instantaneous frequency
  for time-frequency analysis.

## When NOT to use

- **Stationary signals** — Fourier / wavelet cheaper and better.
- **Very short series** — sifting needs enough extrema for
  spline fits.
- **Real-time / online** — EMD is batch; use EEMD / online-EMD
  extensions.
- **When mode-mixing matters** — use Ensemble EMD (Wu-Huang 2009)
  or CEEMDAN to reduce.

## Assumptions & caveats

- **End-point effects** — spline overshoot at boundaries;
  mirror-extend or use edge-effect corrections.
- **Sifting stopping criteria** — Cauchy or S-number-based rules.
- **Mode mixing** — same freq appearing in multiple IMFs;
  EEMD adds noise realisations to average out.
- **No orthogonality** — IMFs not guaranteed orthogonal.

## Related in this repo

- `wavelet-analysis`, `spectral-analysis`,
  `welch-power-spectral-density` — frequency-domain siblings.
- `sarima-arimax`, `arfima`, `arima`, `seasonal-decomposition`,
  `structural-breaks-its`, `change-point-detection` — TS
  decomposition cousins.
- `regime-switching-markov`, `state-space-models`,
  `state-space-kalman` — model-based alternatives.
- `functional-pca`, `functional-basis-smoothing`,
  `functional-linear-model` — functional-data analogues.

## Run

```
python techniques/empirical-mode-decomposition/python/empirical_mode_decomposition.py
Rscript techniques/empirical-mode-decomposition/r/empirical_mode_decomposition.R
```

**Refs:** Huang, N.E. et al. "The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis." *Proc R Soc A* 454(1971): 903-995, 1998; Wu, Z. & Huang, N.E. "Ensemble empirical mode decomposition: a noise-assisted data analysis method." *Adv Adapt Data Anal* 1(1): 1-41, 2009.

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
