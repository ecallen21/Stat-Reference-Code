# Kernel Density Estimation (Reference §6.21)

Smooth, nonparametric estimate of an unknown probability density:

`f̂(x) = (1/(nh)) · Σᵢ K((x − xᵢ)/h)`

where `K` is a **kernel** (symmetric, integrates to 1) and `h > 0` is the **bandwidth**.

## Kernels

| Kernel | `K(u)` | Notes |
|--------|--------|-------|
| Gaussian | `exp(−u²/2)/√(2π)` | Smoothest; the default everywhere |
| Epanechnikov | `0.75(1 − u²)` for `|u| ≤ 1` | Theoretically optimal in MISE; finite support |
| Uniform | `0.5` for `|u| ≤ 1` | Crude; rarely used |
| Triangular | `1 − |u|` for `|u| ≤ 1` | Simple finite-support compromise |
| Cosine | `(π/4)·cos(πu/2)` for `|u| ≤ 1` | Similar smoothness to Epanechnikov |

The **bandwidth `h` matters far more than the kernel**. Two reasonable choices:

- **Silverman's rule of thumb**: `h = 0.9 · min(sd, IQR/1.34) · n^(−1/5)`. Default in R's `density()`.
- **Scott's rule**: `h = sd · n^(−1/5)`. Slightly larger; both come from minimizing AMISE under a normal reference.

Both **oversmooth** clearly bimodal / heavy-tailed data; for those, prefer cross-validation (`bw.ucv`, `bw.bcv` in R) or plug-in (`bw.SJ`).

## Bias–variance tradeoff

- Small `h` → bumpy, high variance, low bias.
- Large `h` → smooth, low variance, high bias.

Try a few bandwidths and look at the resulting plots. There is no universally "correct" `h`.

## Files
- `python/kernel_density_estimation.py` — from-scratch evaluation for five kernels; Silverman/Scott bandwidth helpers; integrates densities to verify they're ≈ 1; compares against `scipy.stats.gaussian_kde`.
- `r/kernel_density_estimation.R` — from-scratch versions + base `stats::density`.
- `pyspark/kernel_density_estimation.py` — `pyspark.mllib.stat.KernelDensity` (Gaussian) with cluster-side Silverman bandwidth computation.

## Run
```
python techniques/kernel-density-estimation/python/kernel_density_estimation.py
Rscript techniques/kernel-density-estimation/r/kernel_density_estimation.R
python techniques/kernel-density-estimation/pyspark/kernel_density_estimation.py
```

**Refs:** Silverman, *Density Estimation for Statistics and Data Analysis*, Chapman & Hall, 1986; Scott, *Multivariate Density Estimation*, 2nd ed., Wiley, 2015.
