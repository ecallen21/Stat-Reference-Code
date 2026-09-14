# WGAN-GP — Wasserstein GAN with Gradient Penalty (Reference §47.286)

Gulrajani, Ahmed, Arjovsky, Dumoulin & Courville (2017).
Improve WGAN training by enforcing the 1-Lipschitz constraint
via a GRADIENT PENALTY instead of weight clipping:

```
L_D = E[D(fake)] − E[D(real)] + λ · E[(‖∇D(x̂)‖ − 1)²]
x̂  = t · real + (1 − t) · fake,   t ~ U[0, 1]
```

Fixes the capacity / stability issues of vanilla WGAN's weight
clipping.

## Files

- `python/wgan_gp_gradient_penalty.py` — 2-layer MLP critic
  and generator with a finite-difference gradient-penalty
  estimator. Demo: 2-D mixture of 3 Gaussians. Initial
  Wasserstein-1 estimate ~1.5; gradient penalty at random x̂
  ~0.7 (target |∇D| = 1). Training loop pseudocode included.
- `r/wgan_gp_gradient_penalty.R` — reticulate + PyTorch WGAN-GP,
  `torch` (R) (R); igul222/improved_wgan_training, tf-gan,
  StudioGAN, from-scratch (Python).

## When to use

- **High-fidelity generation** — WGAN-GP is a robust GAN
  baseline for image / audio / tabular synthesis.
- **When WGAN's clipping causes trouble** — capacity loss,
  gradient explosion; GP is the modern replacement.
- **Comparative benchmarks** — WGAN-GP is often the
  reference in GAN papers.

## When NOT to use

- **Simple / low-dim generation** — a VAE or Gaussian mixture
  may suffice with less tuning.
- **Where diffusion models fit** — modern diffusion + score
  matching often outperforms GANs on image generation.
- **Very small compute** — GP requires a second-order-ish
  gradient calculation each step.

## Assumptions & caveats

- **λ (GP weight)** — 10 is the canonical default.
- **Batch norm in critic** — do NOT use per-sample norms
  (batch norm couples samples); use layer / instance norm.
- **n_critic** (updates per generator step) — 5 is standard.
- **Optimiser** — Adam with β1 = 0, β2 = 0.9 for stability.
- **Mode collapse** still possible; combine with spectral
  normalisation or larger networks.

## Related in this repo

- `wgan-wasserstein-gan` — the weight-clipping baseline.
- `gan-training`, `conditional-gan-cgan` — related GAN
  variants.
- `spectral-normalization` — alternative Lipschitz enforcement.
- `cyclegan-unpaired-translation`,
  `beta-vae-disentangle` — companion generative models.

## Run

```
python techniques/wgan-gp-gradient-penalty/python/wgan_gp_gradient_penalty.py
Rscript techniques/wgan-gp-gradient-penalty/r/wgan_gp_gradient_penalty.R
```

**Refs:** Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V. and Courville, A.C. "Improved training of Wasserstein GANs." In *NeurIPS*, pp. 5769-5779, 2017.

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
