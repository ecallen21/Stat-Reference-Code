# Multitaper Spectral Estimation (Reference §47.332)

Thomson (1982). Average K periodograms of the same series
computed under K discrete prolate spheroidal sequences (DPSS,
aka Slepian tapers):

```
Ŝ(f) = (1/K) Σ_k |Σ_t x_t · w_k(t) · exp(−i 2π f t)|²
```

Trades a small bias for a HUGE variance reduction vs a single-
taper periodogram; standard for geophysics, neuroscience, and
astrophysics spectra.

## Files

- `python/multitaper_spectral_density.py` — Multitaper via
  scipy's DPSS tapers, with periodogram and Welch as baselines.
  Test signal: 60 Hz + 0.5·120 Hz + Gaussian noise. Multitaper
  reduces noise-band PSD std substantially vs periodogram and
  matches Welch on peak identification.
- `r/multitaper_spectral_density.R` — `multitaper`,
  `astsa::mvspec`, `bspec` (R); `scipy.signal.windows.dpss`,
  `mne.time_frequency.psd_multitaper`, `nitime`, `spectrum`,
  from-scratch (Python).

## When to use

- **Neuroscience** — EEG / MEG power spectra, evoked
  responses.
- **Geophysics / astrophysics** — sunspot cycles, seismic
  spectra.
- **Any stationary series** where a periodogram's variance is
  too high.

## When NOT to use

- **Highly non-stationary** — use STFT / wavelets.
- **When frequency resolution is limited by data length** —
  the effective bandwidth is `2 NW / N`.
- **Streaming** — DPSS re-computation per window is costly.

## Assumptions & caveats

- **Stationarity** — Thomson's derivation assumes a stationary
  process.
- **NW parameter** — 3-4 typical; larger = wider bandwidth,
  more tapers.
- **K = 2 NW − 1** — the maximum number of usable tapers.
- **Confidence intervals** — chi-square 2K df at each
  frequency.
- **Jackknife** — Thomson-Chave jackknife variance is standard.

## Related in this repo

- `welch-power-spectral-density`, `spectral-analysis` — the
  Welch / periodogram baselines.
- `hilbert-transform-analytic` — time-domain instantaneous
  view.
- `wavelet-analysis` — time-frequency alternative.
- `dfa-hurst-fluctuation`, `sample-entropy-signal` — related
  time-domain complexity measures.

## Run

```
python techniques/multitaper-spectral-density/python/multitaper_spectral_density.py
Rscript techniques/multitaper-spectral-density/r/multitaper_spectral_density.R
```

**Refs:** Thomson, D.J. "Spectrum estimation and harmonic analysis." *Proc. IEEE*, 70(9): 1055-1096, 1982; Percival, D.B. and Walden, A.T. *Spectral Analysis for Physical Applications*, Cambridge, 1993.

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
