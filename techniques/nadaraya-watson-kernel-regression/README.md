# Nadaraya-Watson Kernel Regression (Reference §5.15)

Nadaraya (1964); Watson (1964). Nonparametric regression estimator
that smooths local averages weighted by a kernel:

    m̂(x) = Σᵢ K((x − xᵢ) / h) · yᵢ  /  Σᵢ K((x − xᵢ) / h)

with `h` the bandwidth and `K` a kernel (Gaussian, Epanechnikov,
uniform).

## Bias-variance

- `h` small → low bias, high variance (wiggly fit).
- `h` large → high bias, low variance (oversmoothed).

## Bandwidth selection

- Silverman rule of thumb: `h = 1.06 · σ_x · n^{−1/5}`.
- Leave-one-out cross-validation (LOO-MSE).
- Plug-in bandwidth (Ruppert-Sheather-Wand).

## Files

- `python/nadaraya_watson_kernel_regression.py` — N-W with Gaussian
  + Epanechnikov kernels, LOO-CV bandwidth from scratch. Demo
  (n=200, true `sin(1.5 x) + 0.3 x`): Silverman h=0.67 too large
  (RMSE 0.25); CV-optimal h=0.26 gives RMSE 0.07.
- `r/nadaraya_watson_kernel_regression.R` — `stats::ksmooth`,
  `KernSmooth::locpoly`, `np::npreg`, `sm::sm.regression` (R);
  `statsmodels.nonparametric.KernelReg` (Python).

## When to use

- **Nonlinear smooth trend with no parametric form in mind**.
- **Exploratory data analysis** — visualise conditional-mean shape.
- **Local weighting** — nearby observations should dominate the
  estimate.

## When NOT to use

- **High dimensions** (p ≥ 3) — curse of dimensionality; use
  additive / spline models.
- **Data with sharp changes** — kernel smoothing over-smooths
  discontinuities; use trees / segmented / change-point.
- **Extrapolation** — kernel estimator is unreliable outside the
  data support.

## Assumptions & caveats

- **Smoothness** of the true regression function.
- **Bandwidth choice** dominates the fit; do CV, not eyeballing.
- **Boundary bias** — the N-W estimator is `O(h)` biased at the
  edges; local-linear (LOESS) is preferred there.
- **Kernel choice** matters less than bandwidth; Epanechnikov is
  asymptotically MSE-optimal.
- **Weighted variance** — CI = `m̂(x) ± z_{α/2} · σ / √(n h · integral K²)`
  needs correcting for bandwidth.

## Related in this repo

- `kernel-density-estimation` — the density analogue.
- `local-regression-loess` — local-linear extension (better at
  boundaries).
- `splines-regression`, `gam`, `gamlss` — global smoothers.
- `varying-coefficient-model` — extends N-W to varying coefficients.

## Run

```
python techniques/nadaraya-watson-kernel-regression/python/nadaraya_watson_kernel_regression.py
Rscript techniques/nadaraya-watson-kernel-regression/r/nadaraya_watson_kernel_regression.R
```

**Refs:** Nadaraya, E.A. "On estimating regression." *Theory of Probability & Its Applications*, 9(1): 141-142, 1964; Watson, G.S. "Smooth regression analysis." *Sankhyā*, 26(4): 359-372, 1964; Wand, M.P. & Jones, M.C. *Kernel Smoothing*, CRC, 1995.

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
