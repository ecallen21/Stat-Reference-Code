# Dynamic Time Warping (Reference §13.22)

Similarity between two time series that may be **misaligned in time**. Euclidean distance is destroyed by even a one-sample shift; DTW finds the optimal monotone alignment path minimizing the sum of pointwise distances.

## Recurrence

```
D[i, j] = |x_i − y_j|
C[0, 0] = D[0, 0]
C[i, j] = D[i, j] + min(C[i−1, j], C[i, j−1], C[i−1, j−1])
```

DTW distance is `C[N, M]`; the optimal alignment path is recovered by backtracking.

## Sakoe-Chiba band

Restrict the alignment to `|i − j| ≤ w` for a window `w`. Two benefits: prevents pathological over-warping and reduces cost from `O(N M)` to `O(N w)`.

## Applications

- **Speech recognition** — the original 1970s use case (Sakoe & Chiba 1978).
- **Gesture / gait / activity classification** from accelerometer streams.
- **Any nearest-neighbor classifier** on time series with local time distortions (see `ts-features-classification` for the kNN-DTW classifier).
- **Time-series alignment** for downstream regression on aligned frames.

## Files

- `python/dynamic_time_warping.py` — from-scratch DTW with Sakoe-Chiba window and alignment-path backtracking. Demos: phase-shifted sinusoids give normalized DTW ≈ 0.05 (vs raw Euclidean 4.2 that ignores the alignment structure); constraint w = 5 runs 27× faster but fits inside the diagonal only when the true warp is small.
- `r/dynamic_time_warping.R` — `dtw::dtw` (rich alignment / step-pattern options and plotting).

## Reporting

- **DTW distance** (raw sum along the optimal path) — depends on path length.
- **Normalized DTW** (divided by path length) — comparable across pairs of different lengths.
- **Warping path** — visualize as a curve on `[0, N] × [0, M]`; near-diagonal = little warping needed.

## Assumptions & caveats

- **Metric?** DTW is a semi-metric (satisfies non-negativity and symmetry but not the triangle inequality). Some downstream algorithms (metric-space indexing) require triangle inequality — use LB_Keogh envelope bounds instead.
- **Endpoint constraint**: requires both sequences to align at both ends. Relaxed variants (Open-DTW, Sub-sequence DTW) allow partial matches.
- **Normalization**: z-normalize each series before DTW when only shape matters.

## Run

```
python techniques/dynamic-time-warping/python/dynamic_time_warping.py
Rscript techniques/dynamic-time-warping/r/dynamic_time_warping.R
```

**Refs:** Sakoe, H. & Chiba, S. "Dynamic programming algorithm optimization for spoken word recognition." *IEEE Trans. Acoust., Speech, Signal Process.* 26(1), 43–49, 1978; Keogh, E. & Ratanamahatana, C.A. "Exact indexing of dynamic time warping." *Knowl. Inf. Syst.* 7(3), 358–386, 2005.

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
