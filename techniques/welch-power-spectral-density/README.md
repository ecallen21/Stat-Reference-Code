# Welch Power Spectral Density (Reference §47.94)

Welch (1967). Estimate the PSD S(f) of a stationary signal by
averaging PERIODOGRAMS on overlapping windowed segments:

    1. Split x into K overlapping (50%) length-L segments.
    2. Apply a window w (Hann default) to reduce leakage.
    3. Average |FFT(w · x_seg)|².

Reduces variance versus a single-shot periodogram (which does NOT
converge in variance as N → ∞); trade frequency resolution
(nperseg) for smoothness.

## Files

- `python/welch_power_spectral_density.py` — from-scratch Welch
  PSD with Hann window + one-sided normalisation. Demo (fs=1000 Hz,
  T=4 s, 60 Hz + 220 Hz tones + noise):
  - nperseg=256 → df=3.9 Hz, peaks detected at (58.6, 62.5, 218.8)
  - nperseg=2048 → df=0.5 Hz, peaks (59.6, 60.1, 220.2), P(60)≈0.5,
    P(220)≈0.125 (Parseval-consistent).
- `r/welch_power_spectral_density.R` — `stats::spectrum`,
  `spectral`, `psd` (R); `scipy.signal.welch`, from-scratch
  (Python).

## When to use

- **Estimating stationary signal PSD** — sensor / speech / vibration.
- **Filter design** — pass / stop band identification.
- **Detecting tonal components** in noisy time series.
- **Bearing / motor fault diagnosis** via characteristic frequencies.

## When NOT to use

- **Non-stationary signals** — use STFT / wavelet / EMD-Hilbert.
- **Very short series** — few segments, poor variance reduction;
  multitaper (Thomson) better.
- **When precise line-tone amplitude matters** — use zero-padded
  DFT + parametric peak fitting.

## Assumptions & caveats

- **Stationarity** required — else PSD is only a local estimate.
- **Window choice** — Hann (default) balances bias / leakage;
  Kaiser tunable, rectangle worst.
- **Overlap** — 50 % standard; more overlap = less variance but
  correlated segments.
- **Normalisation** — one-sided vs two-sided, per-Hz vs per-bin;
  verify with Parseval's identity.

## Related in this repo

- `wavelet-analysis`, `spectral-analysis`,
  `empirical-mode-decomposition` — time-frequency siblings.
- `arima`, `sarima-arimax`, `arfima`, `garch`,
  `stochastic-volatility`, `seasonal-decomposition` — parametric TS
  cousins.
- `realized-volatility-hf` — spectral-adjacent HF-finance analogue.
- `stationarity-tests`, `structural-breaks-its`,
  `change-point-detection` — companion diagnostics.

## Run

```
python techniques/welch-power-spectral-density/python/welch_power_spectral_density.py
Rscript techniques/welch-power-spectral-density/r/welch_power_spectral_density.R
```

**Refs:** Welch, P.D. "The use of fast Fourier transform for the estimation of power spectra: a method based on time averaging over short, modified periodograms." *IEEE Trans Audio Electroacoust* 15(2): 70-73, 1967; Percival, D.B. & Walden, A.T. *Spectral Analysis for Physical Applications.* Cambridge Univ Press, 1993.

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
