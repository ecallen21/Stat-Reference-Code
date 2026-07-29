# Spectral Analysis (Reference §13.18)

Frequency-domain view of a stationary time series. A stationary series can be decomposed into sinusoids at different frequencies; the **spectral density** `f(ω)` describes how variance is distributed across those frequencies. Peaks in `f(ω)` reveal periodicities.

## Raw periodogram

```
I(ω_k) = (1 / T) |Σ_t y_t e^{−iω_k t}|²        ω_k = 2π k / T
```

The periodogram is asymptotically unbiased for `f(ω)` but **inconsistent** — its variance does not shrink with `T`. Two standard remedies:

## Smoothing (Daniell / Bartlett-Priestley kernel)

Running-mean smoother across frequencies. Wider kernel → smoother spectrum but blurred peaks. Trades bias for variance.

## Welch's method

1. Split `y` into overlapping segments of length `nperseg`.
2. Apply a taper (Hann / Hamming) to each segment to reduce spectral leakage.
3. Compute the periodogram of each windowed segment.
4. Average across segments.

Standard implementation: `scipy.signal.welch`.

## Applications

- **Periodicity detection** — daily / weekly / annual cycles.
- **AR/MA diagnostics** — AR(1) with positive φ has monotone-declining `f(ω)`; white noise has flat `f(ω)`.
- **EEG / seismic / audio** — energy-per-band summaries.

## Files

- `python/spectral_analysis.py` — from-scratch raw periodogram, Daniell smoother, and Welch's averaged periodogram with a Hann window. Demo on a signal with sinusoids at periods 20 and 8: Welch estimator matches `scipy.signal.welch` exactly and identifies period ≈ 21.
- `r/spectral_analysis.R` — `stats::spectrum` for raw and Daniell-smoothed periodograms.

## Assumptions

- **Stationarity** — mean and covariance don't drift over the sample window. Non-stationary spectra need time-frequency methods (wavelets, spectrograms).
- **Regular sampling** — irregular timestamps need Lomb-Scargle or explicit interpolation.

## Related methods (deferred)

- **Wavelet analysis** (§13.19), **EMD / HHT** (§13.58), **locally stationary spectra** (§13.59) — for signals whose spectral content evolves over time.
- **Multitaper (Thomson) estimator** — variance reduction via a family of orthogonal tapers.
- **Cross-spectrum / coherence** — spectral analog of the CCF.

## Run

```
python techniques/spectral-analysis/python/spectral_analysis.py
Rscript techniques/spectral-analysis/r/spectral_analysis.R
```

**Refs:** Priestley, M.B. *Spectral Analysis and Time Series.* Academic Press, 1981; Welch, P.D. "The use of Fast Fourier Transform for the estimation of power spectra." *IEEE Trans. Audio Electroacoust.* AU-15(2), 70–73, 1967.

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
