# Butterworth Bandpass Filter (Reference §47.144)

Butterworth (1930). IIR filter with maximally flat magnitude
response in the passband:

    |H(jω)|² = 1 / (1 + (ω/ω_c)^{2n}).

Order-n prototype cascaded as high-pass ∘ low-pass gives a
bandpass. Digital realisation via bilinear transform of the
analogue prototype. Applied forward-and-backward (`filtfilt`) for
zero phase shift.

## Files

- `python/butterworth_bandpass.py` — `scipy.signal.butter` design
  + `filtfilt` zero-phase filtering. 1-kHz, 2-second synthetic signal
  = 2 Hz + 50 Hz + 300 Hz + N(0, 0.1²):
  - Passband [30, 70] Hz: **78.5 %** of retained energy in target
    band; dominant peak at **50.00 Hz** (truth).
  - Passband [0.5, 10] Hz: 40.8 % of energy in band (keeps 2 Hz).
  - Passband [200, 400] Hz: 84.9 % of energy in band (keeps 300 Hz).
- `r/butterworth_bandpass.R` — `signal::butter` + `signal::filtfilt`.

## When to use

- **Biosignal preprocessing** (EEG bandpass 0.5-45 Hz, ECG QRS
  8-20 Hz, EMG 20-450 Hz).
- **Audio, vibration, and communications** front-end filtering.
- Whenever a **maximally flat** magnitude response in the passband
  matters more than sharp roll-off.

## When NOT to use

- **Sharp brick-wall filtering** — Chebyshev / elliptic have steeper
  roll-off (at the price of passband ripple).
- **Real-time / causal** applications — `filtfilt` doubles the
  effective order and introduces future-sample dependency.
- **Very short signals** — filter transients dominate.

## Assumptions & caveats

- **Sampling rate `fs`** and passband edges must satisfy Nyquist:
  0 < f < fs/2.
- **Filter order** trades passband flatness for skirt slope; each
  +order costs +6 dB/oct roll-off.
- **Zero-phase filtering** (`filtfilt`) requires the whole signal
  in memory; not suitable for streaming.
- **Numerical instability** at very low cutoffs; use SOS
  (`sosfiltfilt`) or design a filter in bilinear-mapped z-domain.

## Related in this repo

- `savitzky-golay-filter` — polynomial-LS smoothing alternative.
- `wavelet-denoising`, `hodrick-prescott-filter` — non-linear /
  penalty-based smoothers.
- `spectrogram-analysis`, `welch-psd` — companion spectral
  estimation.

## Run

```
python techniques/butterworth-bandpass/python/butterworth_bandpass.py
Rscript techniques/butterworth-bandpass/r/butterworth_bandpass.R
```

**Refs:** Butterworth, S. "On the theory of filter amplifiers." *Experimental Wireless & the Wireless Engineer* 7, 1930; Oppenheim, A. V. & Schafer, R. W. *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010.

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
