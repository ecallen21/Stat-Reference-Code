# Probabilistic PCA (Reference §25.10)

Tipping & Bishop (1999) — PCA cast as a Gaussian **latent-variable
model**:

```
z_i ~ N(0, I_q)
x_i | z_i ~ N(W z_i + μ, σ² I_d)
```

Closed-form MLE (Tipping-Bishop Thm):

```
W_ML   = U_q (Λ_q − σ² I)^{1/2} R          (any rotation R)
σ²_ML  = (1 / (d − q)) · Σ_{k > q} λ_k     (residual variance)
```

with `(U, Λ)` the eigendecomposition of the sample covariance.

## When to use

- **Missing-value handling** — EM under PPCA is a principled imputer.
- **Model comparison** — the full likelihood enables BIC / DIC.
- **Bayesian PCA** — automatic-relevance-determination selects `q`.
- **When measurement noise is real** — plain PCA gives no noise
  variance.

## When NOT to use

- **Non-Gaussian latent structure** — use ICA / NMF / VAE.
- **Very large d** — closed form needs the full covariance; use
  randomized PCA.
- **Only for visualisation** — plain PCA is fine.

## Files

- `python/probabilistic_pca.py` — closed-form MLE + EM for missing
  data. Demo `d=8, q=3, n=300`: **σ²_ML = 0.251 vs truth 0.25**;
  principal angles between recovered and true subspaces all near 0
  radians. With 15 % missing entries, PPCA-EM imputation RMSE 1.10 vs
  column-mean baseline 1.50 (27 % improvement).
- `r/probabilistic_pca.R` — `pcaMethods` (R Bioconductor);
  `probabilistic-pca`, `sklearn.IterativeImputer` (Python).

## Assumptions & caveats

- **Latent Gaussian** — assumption; residuals should look Gaussian.
- **Isotropic noise σ² I** — factor analysis (FA) relaxes to
  `diag(ψ)` per-feature noise.
- **Number of factors `q`** — profile likelihood, BIC, or Bayesian
  ARD (Bishop 1999).
- **Rotation ambiguity** — `W` identifiable up to right orthogonal
  transform; principal-angle comparison handles it.
- **EM convergence** — closed-form warm start converges in a few
  iterations; deteriorates for high missing fraction.

## Related in this repo

- `dimensionality-reduction-pca` — the point-estimate variant.
- `sparse-pca`, `kernel-pca`, `ica`, `nmf`, `variational-autoencoder`
  — factor-model siblings.
- `exploratory-factor-analysis` — allows anisotropic diagonal noise.
- `missing-data-mi` (if present) — general missing-data alternatives.

## Run

```
python techniques/probabilistic-pca/python/probabilistic_pca.py
Rscript techniques/probabilistic-pca/r/probabilistic_pca.R
```

**Refs:** Tipping, M.E. & Bishop, C.M. "Probabilistic principal component analysis." *JRSS-B*, 1999; Bishop, C.M. "Bayesian PCA." *NeurIPS*, 1999.

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
