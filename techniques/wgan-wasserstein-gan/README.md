# Wasserstein GAN (Reference §47.109)

Arjovsky, Chintala & Bottou (2017). Replaces the JS divergence in
the original GAN with the EARTH-MOVER'S (Wasserstein-1) distance:

    W₁(P, Q) = sup_{|f|_L ≤ 1} E_P[f] − E_Q[f].

Trains a Lipschitz-1 CRITIC via weight clipping (WGAN) or gradient
penalty (WGAN-GP; Gulrajani et al 2017). Empirical benefits:
meaningful loss curve, far less mode collapse, no G/D balancing.

## Files

- `python/wgan_wasserstein_gan.py` — exact 1-D Wasserstein-1 via
  sorted quantiles + JS-on-histograms comparison. Demo (two thin
  N(0, 0.1) distributions sliding apart):
  - shift 0 → W₁ 0.008, JS 0.01
  - shift 1 → W₁ 1.00, JS 0.69 (log 2 saturation)
  - shift 5 → W₁ 5.00, JS 0.69 (**saturated**)
  Wasserstein grows LINEARLY with the shift → informative gradient
  even when supports are disjoint.
- `r/wgan_wasserstein_gan.R` — no first-class R port; `torchGAN`,
  `pytorch-lightning`, `diffusers` in Python.

## When to use

- **Any GAN application** where training instability or mode
  collapse plagued vanilla JS-GAN.
- **Style / attribute transfer** — CycleGAN uses W-GAN losses.
- **When you want a MEANINGFUL loss curve** correlated with sample
  quality.

## When NOT to use

- **Very high-dim samples with weak critic** — clipping / gradient
  penalty can slow learning; try Lipschitz penalty (Petzka 2018).
- **When diffusion models suffice** — modern practice often
  supersedes GANs entirely.

## Assumptions & caveats

- **Lipschitz constraint** — weight clipping is a blunt tool;
  **WGAN-GP** or spectral norm are stronger.
- **Critic capacity** — under-parameterised critic → weak W₁ estimate.
- **Optimiser** — RMSprop for WGAN, Adam for WGAN-GP per paper.
- **Multi-step critic updates** per generator step (n_critic=5 typical).

## Related in this repo

- `gan-training`, `conditional-gan-cgan`, `diffusion-model`,
  `normalizing-flows`, `variational-autoencoder`,
  `energy-based-models` — generative model family.
- `optimal-transport-wasserstein` — the underlying distance.
- `mmd-two-sample-test`, `hsic-independence` — alternative
  distributional losses.
- `spectral-normalization`, `gradient-clipping`,
  `jacobian-regularization` — Lipschitz-related regularisation.

## Run

```
python techniques/wgan-wasserstein-gan/python/wgan_wasserstein_gan.py
Rscript techniques/wgan-wasserstein-gan/r/wgan_wasserstein_gan.R
```

**Refs:** Arjovsky, M., Chintala, S. & Bottou, L. "Wasserstein GAN." *ICML*, 2017; Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V. & Courville, A. "Improved training of Wasserstein GANs." *NeurIPS*, 2017.

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
