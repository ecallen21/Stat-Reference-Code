# Sample Entropy (Reference §47.329)

Richman & Moorman (2000). Signal-complexity measure robust to
short-length bias in ApEn. Given tolerance `r` and template
length `m`:

```
A = # pairs of length-(m+1) sequences within r
B = # pairs of length-m     sequences within r
SampEn(m, r, N) = −log(A / B)
```

Lower values = more regular / self-similar signal; higher =
more random. Common in HRV, EEG, and fault-detection.

## Files

- `python/sample_entropy_signal.py` — Direct O(N²) pairs
  count. Demo with 3 signals of length 500: sine (SampEn ≈
  0), chaotic logistic map (intermediate ≈ 0.6), Gaussian
  noise (highest ≈ 2.3). Tolerance sensitivity: SampEn on
  noise decays from 2.7 (r=0.1σ) to 1.3 (r=0.5σ).
- `r/sample_entropy_signal.R` — `nonlinearTseries`,
  `pracma`, `TSEntropies` (R); `antropy.sample_entropy`,
  `nolds.sampen`, from-scratch (Python).

## When to use

- **HRV / physiologic time series** where regularity matters.
- **EEG / biosignal complexity** — seizure detection,
  anaesthesia monitoring.
- **Fault detection** — vibration signatures becoming more
  or less regular over time.

## When NOT to use

- **Very short series (< 100)** — SampEn variance dominates.
- **Multi-scale phenomena** — use Multiscale Sample Entropy
  (MSE, Costa 2002).
- **Non-stationary series without preprocessing** — remove
  trends first.

## Assumptions & caveats

- **m = 2** is standard; higher m needs more data.
- **r = 0.15-0.25 · σ** is Richman-Moorman's recommendation.
- **Log(0) issue** — if A or B is 0, SampEn is undefined;
  increase tolerance or use fuzzy entropy.
- **Fast implementations** — vectorised pair search or
  k-d tree.

## Related in this repo

- `dfa-hurst-fluctuation` — long-range correlation companion.
- `hilbert-transform-analytic` — instantaneous features.
- `multitaper-spectral-density` — frequency-domain view.
- `change-point-detection`, `ts-anomaly-detection` — related
  monitoring tools.

## Run

```
python techniques/sample-entropy-signal/python/sample_entropy_signal.py
Rscript techniques/sample-entropy-signal/r/sample_entropy_signal.R
```

**Refs:** Richman, J.S. and Moorman, J.R. "Physiological time-series analysis using approximate entropy and sample entropy." *Am. J. Physiol.*, 278(6): H2039-H2049, 2000.

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
