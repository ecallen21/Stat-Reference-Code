# Multidimensional IRT — M2PL (Reference §22.x extra)

Extends 2PL / 3PL to `d ≥ 2` latent traits. Each item has a **discrimination
vector** `a_j ∈ ℝᵈ` (loadings on each trait) and a scalar difficulty `b_j`;
each person has an ability vector `θ_i ∈ ℝᵈ`.

## Compensatory M2PL

```
P(U_ij = 1 | θ_i) = σ( aⱼᵀ θ_i − b_j )
```

High ability on one dimension can compensate for low ability on another.

## Non-compensatory (Sympson)

```
P(U_ij = 1 | θ_i) = Π_k σ( a_{jk} θ_{ik} − b_{jk} )
```

Every dimension must be sufficient — a bottleneck model.

## Fitting

- **Marginal MLE (Bock-Aitkin EM)** — integrate `θ` out against a Gaussian prior via Gauss-Hermite quadrature; maximise the marginal likelihood over `(a, b)`. Consistent as `n → ∞`. `mirt::mirt(..., itemtype='2PL')`.
- **MHRM (Cai 2010)** — Metropolis-Hastings Robbins-Monro; scales beyond `d ≈ 3` where quadrature grids explode.
- **Fully Bayesian** — Stan / JAGS / edstan; report posterior draws for `θ, a, b`.
- **Joint MLE (JML)** — treat `θ` as fixed unknowns. Fast but **inconsistent**; avoid in production.

This module uses a PCA warm-start + per-item logistic regression + EAP `θ`
update via 2-D Gauss-Hermite quadrature — a fast approximation to Bock-Aitkin.

## When to use

- **Attitude / ability tests** that mix content areas (verbal + quantitative).
- **Diagnostic classification** — small `d` with substantive structure.
- **Data-driven dimensionality** — parallel analysis / eigenvalues of the tetrachoric correlation matrix suggest `d`.

## Files

- `python/mirt_multidimensional_irt.py` — PCA warm-start + per-item logistic + EAP update via 2-D Gauss-Hermite quadrature. Demo (n=800, J=20, d=2, planted simple structure): recovered log-lik −7845 (true params LL −8702); Procrustes-aligned loading correlations 0.98 / 0.98; ability correlations 0.82 / 0.84.
- `r/mirt_multidimensional_irt.R` — `mirt::mirt` with `itemtype='2PL'` and multi-factor `model=` spec; `method='MHRM'` for `d > 2`.

## Assumptions & caveats

- **Rotational indeterminacy** — like factor analysis, loadings and `θ` are only identified up to an orthogonal (or oblique) rotation. Report loadings after `varimax`/`oblimin`; interpret rotated factors, not raw axes.
- **Dimensionality choice matters** — under-estimating `d` collapses distinct traits into one; over-estimating leads to weak / redundant factors.
- **Quadrature scales as `n_quad^d`** — 11 nodes in each of 2 dims = 121 grid points; `d = 4` at 11 nodes = ~15000 — use MHRM or SGD-based fitters.
- **Local independence** — items are assumed conditionally independent given `θ`; violated when items share text-stem / testlet structure. Use `mirt` testlet models.
- **JML is inconsistent** — the point estimates in this module are approximations; for hypothesis testing / SEs use `mirt` or a Bayesian fit.

## Run

```
python techniques/mirt-multidimensional-irt/python/mirt_multidimensional_irt.py
Rscript techniques/mirt-multidimensional-irt/r/mirt_multidimensional_irt.R
```

**Refs:** Reckase, M.D. *Multidimensional Item Response Theory*, Springer, 2009; Bock, R.D. & Aitkin, M. "Marginal maximum likelihood estimation of item parameters: application of an EM algorithm." *Psychometrika* 46(4), 443–459, 1981; Cai, L. "High-dimensional exploratory item factor analysis by a Metropolis-Hastings Robbins-Monro algorithm." *Psychometrika* 75(1), 33–57, 2010.

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
