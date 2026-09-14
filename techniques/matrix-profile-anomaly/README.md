# Matrix Profile (Reference §47.290)

Yeh, Zhu, Ulanova et al (2016). For every length-`m` window in
a time series, the matrix profile stores the z-normalised
Euclidean distance to its nearest NON-TRIVIAL match:

```
MP[i] = min_{|j−i| > m} d(T[i:i+m], T[j:j+m])
```

- **Peaks of MP** → DISCORDS (anomalies).
- **Valleys of MP** → MOTIFS (repeating patterns).

O(n²) naive; STOMP / SCRIMP reduce to O(n²) with tiny
constants; STUMPY is the standard implementation.

## Files

- `python/matrix_profile_anomaly.py` — brute-force MP with
  z-normalised windows. Demo: series of length 400 with 3
  planted motif segments and 1 planted step-discord.
  Detected discord at index 292 (planted at 330) with the
  highest MP value; motifs cluster at indices near 50/150/250.
- `r/matrix_profile_anomaly.R` — `tsmp`, `anomalize` (R);
  `stumpy`, `matrixprofile-ts`, `scamp`, from-scratch
  (Python).

## When to use

- **Time-series anomaly detection** on univariate streams
  where subsequences are the anomalies of interest.
- **Motif discovery** — repeating patterns in biosignals,
  IoT, industrial.
- **Preprocessing for retrieval** — MP provides similarity
  primitives.

## When NOT to use

- **Point anomalies** — MP works on SUBSEQUENCES; a single
  outlier is better caught by IQR / z-score.
- **Very long series without STUMPY** — brute-force O(n²) is
  slow; use STOMP / SCRIMP / SCAMP for millions of points.
- **Highly noisy series** — motifs disappear; smooth first.

## Assumptions & caveats

- **Window `m`** — must match the semantic scale of the
  anomaly / motif; tune by domain knowledge.
- **Z-normalisation** — the paper's default; alternative
  (Euclidean without znorm) captures amplitude anomalies too.
- **Trivial-match exclusion** — the window `m` is the
  standard exclusion zone.
- **Multi-scale** — repeat with several `m` values for
  robustness.

## Related in this repo

- `dynamic-time-warping` — the elastic-distance cousin.
- `change-point-detection` — global break-point detection.
- `ts-anomaly-detection` — deep-learning ts-anomaly
  detection.
- `rocket-random-conv-features`, `shapelet-transform` —
  time-series classification cousins.

## Run

```
python techniques/matrix-profile-anomaly/python/matrix_profile_anomaly.py
Rscript techniques/matrix-profile-anomaly/r/matrix_profile_anomaly.R
```

**Refs:** Yeh, C.-C.M., Zhu, Y., Ulanova, L., Begum, N., Ding, Y., Dau, H.A., Silva, D.F., Mueen, A. and Keogh, E. "Matrix profile I: all pairs similarity joins for time series." In *ICDM*, pp. 1317-1322, 2016.

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
