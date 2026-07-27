# Gaussian Mixture Models via EM (Reference §9.12)

Each observation is drawn from one of `K` multivariate Gaussians with unknown mixing proportions:

```
p(x)  =  Σ_k  π_k  N(x | μ_k, Σ_k)
```

Fit via the **EM algorithm**:

- **E-step**: compute posterior *responsibilities* `γ_ik = π_k N(x_i|μ_k,Σ_k) / Σ_j π_j N(x_i|μ_j,Σ_j)`.
- **M-step**: update `π_k = N_k / N`, `μ_k = Σ γ_ik x_i / N_k`, `Σ_k = Σ γ_ik (x_i−μ_k)(x_i−μ_k)' / N_k`.

Guaranteed monotone increase in log-likelihood. Multiple restarts help avoid local optima. Cluster labels come from `argmax_k γ_ik`.

## Choosing K

**BIC**: `−2 log L + p_free · log N`. Fit for a range of K and pick the minimum. AIC is also reported. Free parameters: `K·(p + p(p+1)/2) + (K−1)`.

## GMM vs. k-means

| | k-means | GMM |
|---|---|---|
| Cluster shape | Spherical | Any full-covariance ellipse |
| Assignment | Hard | Soft (responsibilities) |
| Model | Distance minimization | Probability model |
| Handles overlap | No | Yes |
| Model selection | External (silhouette, etc.) | BIC / AIC built-in |

## Files

- `python/gaussian_mixture_models.py` — from-scratch EM with Cholesky-stabilized log-likelihood, multiple restarts, BIC/AIC selection over K. Log-lik and BIC match `sklearn.mixture.GaussianMixture` to 3–4 sig figs.
- `r/gaussian_mixture_models.R` — thin wrapper around `mclust::Mclust`, which is the standard R implementation with 14 covariance parameterizations.
- `pyspark/gaussian_mixture_models.py` — MLlib `GaussianMixture` on distributed data.

## Assumptions

- Data (approximately) generated from a mixture of Gaussians. Non-Gaussian components lead to over-splitting (BIC picks too many components).
- Sample size large enough per component (a rule of thumb: `N / K ≥ 10p`).
- Standardize very-different-scale features first.

## Run

```
python techniques/gaussian-mixture-models/python/gaussian_mixture_models.py
Rscript techniques/gaussian-mixture-models/r/gaussian_mixture_models.R
python techniques/gaussian-mixture-models/pyspark/gaussian_mixture_models.py
```

**Refs:** Dempster, A.P., Laird, N.M. & Rubin, D.B. "Maximum likelihood from incomplete data via the EM algorithm." *JRSS B* 39(1), 1–38, 1977; McLachlan, G.J. & Peel, D. *Finite Mixture Models*, Wiley, 2000; Fraley, C. & Raftery, A.E. "Model-based clustering, discriminant analysis, and density estimation." *JASA* 97(458), 611–631, 2002.

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
