# Gaussian Process Latent Variable Model (Reference §6.16)

Lawrence (2004, NIPS). Nonlinear dimensionality reduction: place a
GP prior on the mapping `X → Y` where `X` is **unknown** (latent,
low-dim), and jointly MLE-optimise `X` + kernel hyperparameters.

## Marginal likelihood

    log p(Y | X, θ) = −½ D log|K + σ² I|
                     − ½ tr((K + σ² I)⁻¹ Y Yᵀ)
                     − constant

Optimise `X` by L-BFGS; usually initialised from PCA.

Linear kernel reduces to **probabilistic PCA**; RBF / Matérn kernels
give nonlinear generalisations.

## Files

- `python/gaussian_process_latent_variable_model.py` — RBF-kernel
  GPLVM with PCA init, L-BFGS optimisation of latent coords from
  scratch. Demo (N=80, D=6, q=2, swiss-roll-ish latent): PCA and
  GPLVM produce similar correlations with the underlying angle —
  richer kernels + Bayesian GPLVM (variational) recover the
  nonlinear structure much more strongly.
- `r/gaussian_process_latent_variable_model.R` — no dedicated
  R package; describes `GPy.models.BCGPLVM`, `gpflow.models.GPLVM`,
  `gpytorch` Bayesian GPLVM (Python).

## When to use

- **Nonlinear low-dim visualisation** — alternative to t-SNE / UMAP
  with a probabilistic model.
- **Missing-data imputation** — GPLVM naturally handles missing
  entries via marginalisation.
- **Uncertainty over latent coordinates** — Bayesian GPLVM gives
  posterior over `X`.
- **Interpretable manifold assumption** — GP smoothness prior beats
  discrete embeddings for downstream Gaussian modelling.

## When NOT to use

- **Very large N** — `O(N³)` per iteration; use sparse / variational
  variants (Titsias 2010).
- **Discrete data** — extend with binomial / Poisson likelihoods (GLM-
  GPLVM) or use topic models.
- **You only need clusters** — t-SNE / UMAP / spectral clustering may
  visualise better.

## Assumptions & caveats

- **Init sensitivity** — PCA init helps; multiple restarts advisable.
- **Kernel choice** — RBF is default; Matérn / additive kernels for
  more structure.
- **Regularisation** — `σ²` (noise) and lengthscale trade explanation
  vs smoothing; jointly optimise.
- **Non-identifiability** — orthogonal rotations of `X` give the same
  likelihood; report distances or Procrustes-align.

## Related in this repo

- `pca`, `probabilistic-pca`, `sparse-pca` — linear siblings.
- `isomap`, `lle-locally-linear-embedding`, `diffusion-maps`,
  `tsne-umap` — nonlinear dim-red neighbours.
- `gaussian-process-regression` — the base GP model.

## Run

```
python techniques/gaussian-process-latent-variable-model/python/gaussian_process_latent_variable_model.py
Rscript techniques/gaussian-process-latent-variable-model/r/gaussian_process_latent_variable_model.R
```

**Refs:** Lawrence, N.D. "Gaussian process latent variable models for visualisation of high-dimensional data." *NIPS*, 2004; Titsias, M. & Lawrence, N.D. "Bayesian Gaussian process latent variable model." *AISTATS*, 2010.

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
