# Generative Adversarial Network — GAN (Reference §27.9)

Two neural networks in an adversarial minimax game (Goodfellow et al. 2014):

- **Generator** `G_θ: ℝ^k → ℝ^d` — turns noise `z ~ p(z)` into fake data.
- **Discriminator** `D_φ: ℝ^d → [0, 1]` — probability that the input is real.

## Losses

Standard (Goodfellow original):

```
L_D = − 𝔼_{x ~ p_data} [log D(x)]  −  𝔼_{z ~ p(z)} [log(1 − D(G(z)))]
L_G =  𝔼_{z ~ p(z)} [log(1 − D(G(z)))]
```

**Non-saturating** (used in the demo — better gradients for G early in training):

```
L_G = − 𝔼_{z ~ p(z)} [log D(G(z))]
```

Under the minimax game, the optimal discriminator is `D*(x) = p_data(x) / (p_data(x) + p_G(x))`; substituting gives the JS-divergence, which G minimises.

## Wasserstein GAN and beyond

- **WGAN / WGAN-GP** (Arjovsky, Gulrajani) — drop the sigmoid, use Earth-Mover distance, add gradient penalty; more stable training, no vanishing gradients when D is confident.
- **Spectral normalisation** — cheaper Lipschitz regularisation than WGAN-GP.
- **StyleGAN, BigGAN, progressive-growth** — production-scale image GANs.
- **Modern default**: diffusion models (DDPM, EDM, flow-matching) — better sample quality at higher compute.

## When to use

- **Fast unconditional sampling** — GANs decode in one forward pass (vs many for diffusion).
- **Image translation / super-resolution / inpainting** — conditional GANs (pix2pix, CycleGAN).
- **Latent-space traversal** — StyleGAN's disentangled latents.
- **Privacy-preserving synthetic data** — CTGAN and family for tabular data.

## Files

- `python/gan_training.py` — from-scratch numpy GAN with MLP G and D, non-saturating loss, alternating SGD. Demo on the classic **ring of 8 Gaussians** (N=800): after 4000 iterations, G's samples land in all 8 modes (27–73 samples each) — no mode collapse. Sample mean matches data mean.
- `r/gan_training.R` — `torch::nn_module` for G and D, `nnf_binary_cross_entropy_with_logits`; `keras3` with two Sequential models and a custom `train_step`.

## Assumptions & caveats

- **Mode collapse** — G may cover only some modes of `p_data`. Mitigations: minibatch discrimination, unrolled GANs, WGAN-GP, feature-matching loss.
- **Training instability** — the minimax game oscillates; balance `k_disc` (discriminator steps per generator step), use Adam(betas=(0.5, 0.999)), spectral norm on D.
- **Evaluation is hard** — no likelihood; use FID, IS, precision/recall of samples vs data, or downstream classifier accuracy.
- **No test likelihood** — GANs are implicit; you can't compute `p_G(x)`. Explicit alternatives: flows, autoregressive models, diffusion.
- **Sensitive to hyperparameters** — lr, betas, batch size, latent dim, activation. Reproducibility across papers has been a known issue (Lucic 2018).
- **Discriminator overfits to fakes** as training proceeds; label smoothing (positives labelled 0.9) helps.

## Related in this repo

- `autoencoder`, `variational-autoencoder` — likelihood-based generative alternatives.
- `dirichlet-process-mixture`, `bayesian-hierarchical-models` — probabilistic mixture generatives.
- `adam-optimizer`, `dropout-batchnorm` — training add-ons standard in modern GANs.

## Run

```
python techniques/gan-training/python/gan_training.py
Rscript techniques/gan-training/r/gan_training.R
```

**Refs:** Goodfellow, I. et al. "Generative adversarial nets." *NeurIPS*, 2014; Arjovsky, M., Chintala, S. & Bottou, L. "Wasserstein GAN." *ICML*, 2017; Gulrajani, I. et al. "Improved training of Wasserstein GANs." *NeurIPS*, 2017; Karras, T. et al. "Analyzing and improving the image quality of StyleGAN." *CVPR*, 2020.

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
