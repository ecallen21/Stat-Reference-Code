# Epanechnikov Kernel Density (Reference §47.388)

Epanechnikov (1969). Kernel

```
K(u) = 0.75 · (1 − u²)   for |u| ≤ 1
     = 0                  otherwise
```

MINIMISES asymptotic mean integrated squared error (AMISE)
among all non-negative kernels — the ONLY optimal-MISE kernel
in the Epanechnikov class. Efficiency vs Gaussian ≈ 94 % ⇒
Gaussian is nearly optimal, but Epanechnikov's COMPACT SUPPORT
makes it faster and cleaner to plot (no long tails).

KDE:

```
f̂_h(x) = (1 / (n · h)) · Σ_i K((x − X_i) / h)
```

## Files

- `python/epanechnikov_kernel_density.py` — bimodal mixture
  `0.5 N(−2, 1) + 0.5 N(2, 1.5²)`, n=200 samples. Silverman
  rule-of-thumb bandwidth 0.84. Integrated squared error
  vs the truth: Epanechnikov 0.0028, Gaussian 0.0038 —
  Epanechnikov slightly ahead on this sample.
- `r/epanechnikov_kernel_density.R` — `density(kernel='epanechnikov')`
  (base R), `KernSmooth::bkde`, `ks::kde` (R);
  `sklearn.neighbors.KernelDensity(kernel='epanechnikov')`,
  from-scratch (Python).

## When to use

- **Univariate density estimation** — histograms are noisy;
  KDE is smoother and consistent.
- **When compact support matters** — plots that clip
  cleanly at data range, e.g. survival-time densities.
- **Foundation for higher-dim KDE** — tensor product of
  Epanechnikov kernels.

## When NOT to use

- **Very small n** — bandwidth selection is unstable; use
  parametric models.
- **High-dim** — curse of dimensionality; use local density
  neighbourhoods (kNN) or projection (probabilistic PCA).
- **Boundary regions** — kernel puts mass outside the
  support; use boundary-corrected kernels or logspline.

## Assumptions & caveats

- **Bandwidth selection** — Silverman's rule `1.06 σ n^(−1/5)`
  is a rule-of-thumb; cross-validation (LSCV / SJPI) or
  plug-in methods give tighter fits.
- **Optimal-MISE derivation** — Epanechnikov's theorem
  assumes twice-differentiable target density and
  minimises AMISE over kernels with mean 0 and finite
  variance.
- **Curse of dimensionality** — AMISE scales `O(n^(−4/(d+4)))`.
- **Non-negativity in the kernel** — enables the KDE to be a
  valid density.
- **Higher-order (bias-reduced) kernels** — sacrifice non-
  negativity for lower bias; not for plotting.

## Related in this repo

- `kernel-density-estimation` — general KDE (may reference
  this file for the optimal-MISE kernel).
- `nadaraya-watson-kernel-regression`,
  `local-regression-loess`, `functional-basis-smoothing` —
  kernel-smoothing cousins.
- `splines-regression`, `additive-quantile-regression` —
  alternative smoothers.
- `histogram-binning-calibration` — histogram cousin.

## Run

```
python techniques/epanechnikov-kernel-density/python/epanechnikov_kernel_density.py
Rscript techniques/epanechnikov-kernel-density/r/epanechnikov_kernel_density.R
```

**Refs:** Epanechnikov, V.A. "Non-parametric estimation of a multivariate probability density." *Theory of Probability & Its Applications*, 14(1): 153-158, 1969.

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
