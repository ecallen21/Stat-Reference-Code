# Functional Basis Smoothing (Reference §31.1)

Represent a curve `x(t)` in a **basis expansion**:

```
x(t) = Σ_j c_j φ_j(t).
```

Common bases: **Fourier** (periodic), **B-splines** (local support),
**wavelets** (non-stationary features).

## P-spline smoothing (Eilers-Marx 1996)

Penalise the second differences of basis coefficients:

```
ĉ = argmin_c  Σ_i (y_i − Φ(t_i)ᵀ c)²  +  λ · cᵀ R c
   = (ΦᵀΦ + λ R)⁻¹ Φᵀ y
```

`R = D₂ᵀ D₂` with `D₂` the second-difference matrix.

## When to use

- **Any noisy curve** you want to smooth without over-fitting.
- **Pre-processing** step before FPCA / functional regression /
  clustering.
- **Interpolation** at arbitrary grid points.

## When NOT to use

- **Sharp jumps** — smoothing blurs them; use TV / fused-LASSO.
- **Extreme extrapolation** — splines are unreliable beyond the data
  support.

## Files

- `python/functional_basis_smoothing.py` — from-scratch cubic B-spline
  (truncated-power basis) + second-order roughness penalty + LOO-CV.
  Demo on `sin(2πt) + 0.5t + noise`: **CV-optimal λ = 0.0001**;
  **raw MSE 0.083 → smoothed MSE 0.006 (93 % reduction)**.
- `r/functional_basis_smoothing.R` — `fda::smooth.basis`,
  `mgcv::gam(bs='ps')`, `splines::bs` (R); `scikit-fda`, `patsy`
  (Python).

## Assumptions & caveats

- **Number of knots** — moderately large (10-40); over-fit is
  controlled by the penalty.
- **Order of penalty** — 2 (piecewise linear ideal) is standard; 1 or
  3 for different behaviour at boundaries.
- **Boundary effects** — natural cubic splines behave linearly outside
  the data.
- **GCV** — generalised CV (Wahba 1978) is a common alternative to
  LOO-CV.
- **Fourier basis** — preferable when the data are periodic.

## Related in this repo

- `functional-pca`, `functional-regression`, `functional-anova`,
  `functional-clustering`, `curve-registration`, `functional-depth`,
  `functional-linear-model` — the FDA family.
- `local-regression-loess`, `kernel-density-estimation` — kernel
  smoothing siblings.
- `restricted-cubic-splines` (if present) — natural-spline sibling.
- `wavelet-analysis` — non-stationary alternative.

## Run

```
python techniques/functional-basis-smoothing/python/functional_basis_smoothing.py
Rscript techniques/functional-basis-smoothing/r/functional_basis_smoothing.R
```

**Refs:** Eilers, P.H.C. & Marx, B.D. "Flexible smoothing with B-splines and penalties." *Statistical Science*, 1996; Ramsay, J.O. & Silverman, B.W. *Functional Data Analysis*, Springer, 2005 (Ch. 4-5); Wahba, G. "Smoothing noisy data with spline functions." *Numer Math*, 1975.

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
