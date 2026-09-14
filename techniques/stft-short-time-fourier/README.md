# Short-Time Fourier Transform (Reference §47.361)

Allen & Rabiner (1977); Portnoff (1976); Griffin-Lim (1984)
for phase reconstruction. Slice the signal into overlapping
WINDOWED segments and take the FFT of each:

```
X(k, m) = Σ_n x[n] · w[n − mH] · exp(−j·2π·k·n / N)
```

- `w` — analysis taper (Hann is the default; Hamming, Kaiser,
  Blackman available)
- `N` — FFT length; determines frequency resolution `fs/N`
- `H` — hop size; determines time resolution and overlap

Produces a time-frequency (spectrogram) representation.
Overlap-add (COLA) reconstruction returns the original signal
to machine precision when `Σ_m w²[n − mH] ≡ 1`.

## Files

- `python/stft_short_time_fourier.py` — Linear chirp
  50 → 250 Hz at fs = 1000 Hz over 2 s (2 000 samples). STFT
  with N=256, hop=64, Hann window (28 frames × 129 bins).
  Peak-bin tracking recovers the instantaneous frequency to
  ±2 Hz. Overlap-add reconstruction error 1.4e-16 (machine
  precision).
- `r/stft_short_time_fourier.R` — `seewave::stft`,
  `signal::specgram`, `sound::spectrogram` (R);
  `scipy.signal.stft`/`istft`, `librosa.stft`,
  `torchaudio.transforms.Spectrogram`, from-scratch (Python).

## When to use

- **Speech / audio** — MFCC pipelines, source separation,
  audio classification.
- **Vibration / seismic / EEG monitoring** — non-stationary
  spectra.
- **Music information retrieval** — pitch, onset, chord.
- **Time-frequency masking / phase-vocoder** — audio effects.

## When NOT to use

- **Very short transients** — STFT smears in time; use
  wavelets or matching-pursuit for sparse time-frequency
  atoms.
- **Ultra-fine frequency resolution** — multitaper or
  Prony's method reduce estimator variance without more
  bandwidth.
- **When phase matters and only magnitude is available** —
  Griffin-Lim iterative phase estimation, or explicit
  complex-valued STFT.

## Assumptions & caveats

- **Window choice** — trade-off between main-lobe width
  (frequency resolution) and side-lobe attenuation (leakage);
  Hann/Hamming/Blackman/Kaiser cover common trade-offs.
- **COLA condition** — hop and window must satisfy
  `Σ_m w²[n − mH] ≡ const` for perfect reconstruction; hop
  = N/4 with Hann is standard.
- **Frequency resolution** — Δf = fs / N; larger N ⇒ finer
  frequency but coarser time.
- **Complex output** — magnitude for perception, phase for
  reconstruction; unwrapping phase is non-trivial.
- **Log-magnitude / dB** — usual visualisation is on a
  log-magnitude scale.

## Related in this repo

- `spectral-analysis`, `welch-power-spectral-density`,
  `multitaper-spectral-density`, `hilbert-transform-analytic`
  — spectral cousins.
- `wavelet-analysis`, `empirical-mode-decomposition` —
  time-frequency alternatives.
- `speech-recognition-whisper`, `wav2vec-ssl-audio`,
  `audio-diffusion-audioldm` — modern audio-DL neighbours.
- `savitzky-golay-filter`, `butterworth-bandpass` —
  time-domain filtering complements.

## Run

```
python techniques/stft-short-time-fourier/python/stft_short_time_fourier.py
Rscript techniques/stft-short-time-fourier/r/stft_short_time_fourier.R
```

**Refs:** Allen, J.B. and Rabiner, L.R. "A unified approach to short-time Fourier analysis and synthesis." *Proc. IEEE*, 65(11): 1558-1564, 1977; Griffin, D. and Lim, J. "Signal estimation from modified short-time Fourier transform." *IEEE Trans. ASSP*, 32(2): 236-243, 1984.

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
