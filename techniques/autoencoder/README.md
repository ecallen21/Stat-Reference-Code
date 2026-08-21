# Autoencoder (Reference §27.7)

Unsupervised representation learner:

```
encoder e_φ: ℝ^d → ℝ^k    (bottleneck, k << d)
decoder d_θ: ℝ^k → ℝ^d
L(θ, φ) = || x − d_θ(e_φ(x)) ||²        (reconstruction MSE)
```

## Linear vs non-linear

- **Linear autoencoder + squared loss** = PCA (up to rotation).
- **Non-linear** (ReLU / GELU / tanh) can learn richer manifolds; harder to train; local optima.
- **Deep** (stacked, tied weights) — historical baseline before end-to-end pretraining.

## Denoising autoencoder (Vincent 2008)

Feed a **corrupted** input `x̃ = x + ε` and reconstruct the **clean** `x`:

```
L = || x − d(e(x̃)) ||²
```

Forces the encoder to learn robust features that ignore noise. Related in
spirit to masked-language-model pretraining (BERT masks tokens, reconstructs).

## When to use

- **Dimensionality reduction** — non-linear alternative to PCA.
- **Denoising / imputation** — restore corrupted or missing entries.
- **Anomaly detection** — a large reconstruction error flags an out-of-distribution input.
- **Pretraining** for downstream tasks (largely superseded by contrastive / masked-token methods).
- **Feature learning** before a downstream classifier / clusterer.

## Related architectures

- **Variational autoencoder** — probabilistic latent + regularised bottleneck (see `variational-autoencoder`).
- **Sparse autoencoder** — L1 penalty on latent activations.
- **Contractive autoencoder** — Frobenius-norm penalty on encoder Jacobian.
- **Masked autoencoder** (He 2022) — mask random patches of input, reconstruct only masked ones; ViT-based pretraining.

## Files

- `python/autoencoder.py` — from-scratch non-linear AE + denoising variant, trained by manual SGD backprop. Demo (400 samples from a rank-3 subspace in ℝ¹⁰ + iid noise): after 2000 epochs, non-linear AE with `k=3` bottleneck reaches MSE 0.0094 vs PCA-optimal 0.0068 (near-optimal reconstruction); denoising AE with input-noise σ=0.3 reaches MSE 0.011 — small penalty for robustness to noise.
- `r/autoencoder.R` — `torch::nn_sequential`, `keras3` encoder + decoder Sequential, `h2o::h2o.deeplearning(autoencoder=TRUE)`.

## Assumptions & caveats

- **Under- vs over-complete** — `k < d` forces compression; `k ≥ d` risks learning identity unless regularised.
- **Non-linear AE with squared loss on Gaussian data doesn't beat PCA in the linear-encoder sublimit** — most of the gain comes from non-linear manifold data (images, text).
- **Local optima** — random init + SGD can converge to a much worse solution than PCA on low-dim linear data; multiple restarts or a linear-warmup phase help.
- **Reconstruction MSE is a proxy for downstream quality** — a low reconstruction loss doesn't guarantee useful representations. Evaluate on the downstream task.
- **Anomaly detection with MSE** works only when the anomaly is truly out-of-manifold; noise-heavy tails inside the manifold don't reconstruct poorly.
- **Denoising AE** with heavy noise learns essentially the same manifold projection as PCA, but only if the manifold is locally linear near each point.

## Related in this repo

- `pca`, `kernel-pca`, `independent-components` — linear / kernel counterparts.
- `variational-autoencoder` — probabilistic generalisation.
- `gan-training` — implicit generative alternative.
- `dropout-batchnorm`, `adam-optimizer` — training add-ons.
- `isolation-forest-anomaly` — non-neural alternative for anomaly detection.

## Run

```
python techniques/autoencoder/python/autoencoder.py
Rscript techniques/autoencoder/r/autoencoder.R
```

**Refs:** Hinton, G.E. & Salakhutdinov, R.R. "Reducing the dimensionality of data with neural networks." *Science* 313(5786), 504–507, 2006; Vincent, P. et al. "Extracting and composing robust features with denoising autoencoders." *ICML*, 2008; He, K. et al. "Masked autoencoders are scalable vision learners." *CVPR*, 2022.

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
