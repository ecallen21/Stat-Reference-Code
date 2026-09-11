# Latent Diffusion — LDM / Stable Diffusion (Reference §47.211)

Rombach, Blattmann, Lorenz, Esser & Ommer (2022, CVPR — aka Stable
Diffusion). Two stages:

1. **Train a VAE** that compresses 512 × 512 images to 64 × 64
   latents.
2. **Train a diffusion model in this latent space** (not pixel
   space).

Diffusion in a 64× compressed latent is ~64× cheaper per step
while a good VAE preserves visual quality. Enabled consumer-GPU
text-to-image generation.

## Files

- `python/latent_diffusion_ldm.py` — toy LDM with linear (SVD)
  VAE + oracle 'diffusion' step. 256-D data on 16-D manifold:
  - VAE reconstruction relative error: **0.011** (16× compression).
  - After latent-space diffusion + decode: **rel error 0.013**.
  - Diffusion FLOPs at 20 steps: **256× cheaper** in latent
    space (20,480 vs 5.24 M for the toy).
- `r/latent_diffusion_ldm.R` — no R port; recommends
  `diffusers.StableDiffusionPipeline`, `AutoencoderKL`.

## When to use

- **High-resolution generation** where pixel-space diffusion is
  infeasible.
- **Text-to-image / image-to-image / inpainting** with
  pretrained VAE + UNet.
- **Fine-tuning to a domain** — cheaper than training pixel-
  space from scratch.

## When NOT to use

- **Low-resolution data** (32 × 32) — the VAE overhead outweighs
  the savings.
- **VAE-sensitive tasks** — the compressor can lose text /
  fine detail; pixel-space is safer.
- **When exact pixel fidelity matters** (medical imaging) — VAE
  is lossy.

## Assumptions & caveats

- **VAE quality** dominates the ceiling; SD's f8 VAE (8× spatial,
  4× channel = 32× overall) is a hard-earned trade.
- **Latent-space priors** are Gaussian-adjacent; a bad VAE
  produces off-manifold latents that the diffusion model can't
  denoise.
- **Text conditioning** via cross-attention (CLIP encoder).
- **KL vs discrete VQ**: SD uses continuous KL-VAE; VQ-Diffusion
  uses discrete codes.

## Related in this repo

- `diffusion-model`, `ddim-implicit-diffusion`,
  `classifier-free-guidance`, `score-based-sde`,
  `rectified-flow`, `consistency-models` — diffusion-family
  neighbours.
- `variational-autoencoder`,
  `autoencoder` — VAE building blocks.
- `controlnet-conditional` — SD conditioning cousin.

## Run

```
python techniques/latent-diffusion-ldm/python/latent_diffusion_ldm.py
Rscript techniques/latent-diffusion-ldm/r/latent_diffusion_ldm.R
```

**Refs:** Rombach, R., Blattmann, A., Lorenz, D., Esser, P. & Ommer, B. "High-resolution image synthesis with latent diffusion models." *CVPR*, 2022; Kingma, D. P. & Welling, M. "Auto-encoding variational Bayes." *ICLR*, 2014.

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
