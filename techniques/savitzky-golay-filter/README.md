# Savitzky-Golay Filter (Reference §47.142)

Savitzky & Golay (1964). Sliding-window polynomial least-squares
smoother: within each window of length w = 2m + 1, fit a degree-p
polynomial by OLS and replace the centre with the fitted value at
0. Equivalent to a fixed-coefficient FIR convolution that depends
only on (window, poly-order, derivative-order); preserves higher
moments of the signal better than a plain moving average.

## Files

- `python/savitzky_golay_filter.py` — from-scratch coefficient
  computation via the pseudo-inverse of the Vandermonde matrix.
  Test signal = sin(t) + 0.3 sin(3t) + N(0, 0.3²) on 400 samples:
  - Raw RMSE (no filter): 0.299
  - w = 11, p = 3: RMSE = **0.134**
  - w = 21, p = 3: RMSE = **0.100**
  - w = 51, p = 3: RMSE = **0.074** (over-smoothed for sharp
    features).
  - First-derivative estimation (w = 31, p = 4, deriv = 1) achieves
    RMSE ≈ 0.58 vs analytic dy/dt.
- `r/savitzky_golay_filter.R` — `signal::sgolayfilt` for
  smoothing / differentiation; `prospectr::savitzkyGolay` common in
  chemometrics.

## When to use

- **Smoothing spectra** (Raman, NIR, IR, mass-spec) while preserving
  peak height / width.
- **Numerical differentiation** of noisy signals (deriv = 1, 2).
- **Trend estimation** where high-frequency noise must go but
  moment-preservation matters.

## When NOT to use

- **Strongly non-stationary** signals where polynomial-order fit
  breaks down.
- **Signals with sharp discontinuities** — SG smears edges; use
  wavelet / median filter instead.
- **Very short signals** (< a few window lengths) — edge effects
  dominate.

## Assumptions & caveats

- **Odd window length** required; center point is the fitted value.
- **poly < window** — otherwise fit is a perfect interpolant and
  no smoothing happens.
- **Derivative estimates** divide by (dt)ᵈ to get physical units.
- **Edge handling** — mirror / reflect padding here; scipy's
  `savgol_filter(mode="mirror")` matches this behaviour.

## Related in this repo

- `functional-basis-smoothing` — global polynomial / spline
  smoothing.
- `butterworth-bandpass`, `wavelet-denoising` — frequency-domain
  alternatives.
- `local-regression-loess`, `kernel-smoothing` — non-parametric
  local smoothers.

## Run

```
python techniques/savitzky-golay-filter/python/savitzky_golay_filter.py
Rscript techniques/savitzky-golay-filter/r/savitzky_golay_filter.R
```

**Refs:** Savitzky, A. & Golay, M. J. E. "Smoothing and differentiation of data by simplified least-squares procedures." *Analytical Chemistry* 36(8), 1964; Schafer, R. W. "What is a Savitzky-Golay filter?" *IEEE Signal Processing Magazine* 28(4), 2011.

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
