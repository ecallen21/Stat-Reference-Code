# Hilbert Transform / Analytic Signal (Reference §47.331)

Hilbert (1912); Gabor (1946). Construct the analytic signal:

```
z(t) = x(t) + i · H(x)(t)
```

From `z` one reads instantaneous amplitude `|z|`, phase
`arg z`, and frequency `d(arg z)/dt / (2π)`.

## Files

- `python/hilbert_transform_analytic.py` — FFT-based analytic
  signal. AM demo: 50 Hz carrier with envelope `1 + 0.5 t`;
  the demodulated `A(t)` tracks the true envelope. Chirp demo
  (20 → 80 Hz): instantaneous frequency estimate matches
  `20 + 30 t` to within 0.1 Hz.
- `r/hilbert_transform_analytic.R` — `signal::hilbert`,
  `seewave::env`, `hht` (R); `scipy.signal.hilbert`,
  `emd-signal`, from-scratch (Python).

## When to use

- **Envelope detection** — AM demodulation, amplitude tracking
  in vibration signals.
- **Instantaneous frequency** — chirp analysis, EEG frequency
  drift.
- **Phase-based analysis** — phase-locking, coherence.
- **Hilbert-Huang transform** — with Empirical Mode
  Decomposition for non-stationary data.

## When NOT to use

- **Signals with multi-band content** — instantaneous
  frequency is ill-defined; use STFT or wavelets.
- **Very short signals** — FFT-based Hilbert has boundary
  artefacts.
- **DC / near-DC signals** — the Hilbert transform zeros the
  DC.

## Assumptions & caveats

- **Bedrosian theorem** — Hilbert factorises A(t)·cos(φ(t))
  only when A and cos have disjoint spectra.
- **Boundary effects** — mirror-pad or discard edges.
- **Envelope + phase artefacts** — a NARROWBAND assumption
  helps interpretability.
- **Real-time processing** — the FFT-based version is
  non-causal; use IIR Hilbert for streaming.

## Related in this repo

- `wavelet-analysis` — multiresolution alternative.
- `sample-entropy-signal`, `dfa-hurst-fluctuation` — related
  signal descriptors.
- `multitaper-spectral-density` — companion frequency-domain
  estimator.
- `welch-power-spectral-density` — classical spectral tool.

## Run

```
python techniques/hilbert-transform-analytic/python/hilbert_transform_analytic.py
Rscript techniques/hilbert-transform-analytic/r/hilbert_transform_analytic.R
```

**Refs:** Gabor, D. "Theory of communication." *J. IEE*, 93(26): 429-457, 1946; Marple, S.L. "Computing the discrete-time analytic signal via FFT." *IEEE Trans. Signal Process.*, 47(9): 2600-2603, 1999.

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
