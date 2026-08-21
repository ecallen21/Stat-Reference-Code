# Denoising Diffusion Probabilistic Model — DDPM (Reference §27.x extra)

Ho, Jain & Abbeel (2020). Generative model that learns to reverse a fixed
noise-adding process.

## Forward process (fixed)

Add Gaussian noise across `T` steps to data `x_0`:

```
x_t = √(ᾱ_t) x_0 + √(1 − ᾱ_t) · ε,   ε ~ N(0, I)
ᾱ_t = Π_{s ≤ t} (1 − β_s)
```

Typical schedule: `β_1 ≈ 10⁻⁴`, linearly to `β_T ≈ 0.02` at `T = 1000`.

## Reverse process (learned)

A neural denoiser `ε_θ(x_t, t)` predicts the noise that was added:

```
L_simple = 𝔼_{t, x_0, ε} ‖ ε − ε_θ(x_t, t) ‖²
```

Training is one forward + one denoise + MSE. No adversary, no likelihood
computation.

## Sampling

```
x_{t-1} = (1 / √α_t) · [ x_t − (1 − α_t) / √(1 − ᾱ_t) · ε_θ(x_t, t) ] + σ_t · z
```

`z ~ N(0, I)` for `t > 0`, no noise at `t = 0`. Standard: `σ_t² = β_t` (DDPM)
or a learned schedule.

## Family

- **DDPM** (Ho 2020) — the original.
- **DDIM** (Song 2020) — deterministic sampler; 10–50 steps instead of 1000.
- **Score-based / EDM** (Karras 2022) — SDE view; SOTA image quality.
- **Latent diffusion / Stable Diffusion** (Rombach 2022) — diffuse in a VAE latent for speed.
- **Flow matching** (Lipman 2023) — straight-line probability paths; simpler ODE.
- **Consistency models** (Song 2023) — single-step samplers via distillation.

## When to use

- **High-quality generation** — images, audio, molecules, protein structures.
- **Conditional generation** — text-to-image (Stable Diffusion, DALL-E 3, Imagen), text-to-audio (AudioLDM), text-to-3D.
- **Inverse problems** — inpainting, deblurring, super-resolution via guided diffusion.
- **Molecular design** — geometric diffusion for drug discovery.
- **Not real-time** — even fast schedulers need 4–20 forward passes; consistency-distilled models get to 1–2 steps.

## Files

- `python/diffusion_model.py` — from-scratch numpy DDPM with a tiny MLP denoiser and manual back-prop. Linear β schedule with `T = 40` steps; trained for 1500 epochs on the classic **two-moons** 2-D dataset. Demo: sample statistics match data (mean [0.32, 0.22] vs [0.48, 0.23]; sd [0.92, 0.48] vs [0.92, 0.49]); 57% of generated samples land within radius 0.15 of a training point — good manifold coverage.
- `r/diffusion_model.R` — `torch` (manual DDPM), `reticulate` + `huggingface diffusers`, `denoising-diffusion-pytorch`.

## Assumptions & caveats

- **`T` and `β` schedule** matter — cosine or sigmoid schedules (Nichol-Dhariwal 2021) beat linear at `T ≈ 100+`.
- **Compute** — training a real image diffusion model needs thousands of GPU-hours; sampling is 10–50 forward passes.
- **Distillation** cuts sampling to 1–4 steps (consistency, progressive distillation, DMD).
- **Classifier-free guidance** (Ho-Salimans 2022) is the standard trick for text-to-image quality; a guidance scale ~3–7 trades diversity for fidelity.
- **VAE-latent diffusion** (Rombach 2022) is 100× cheaper than pixel-space diffusion at similar quality — the recipe behind Stable Diffusion.
- **Diffusion for continuous data** — for discrete (text, categorical) data use masked diffusion / D3PM / discrete flow-matching.

## Related in this repo

- `variational-autoencoder`, `gan-training` — alternative generative families.
- `autoencoder` — the latent-space substrate used in latent diffusion.
- `attention-mechanism`, `transformer-encoder`, `convolutional-nn`, `residual-connections` — building blocks of the U-Net / DiT denoiser.
- `lr-schedules`, `adam-optimizer` — training-loop essentials.

## Run

```
python techniques/diffusion-model/python/diffusion_model.py
Rscript techniques/diffusion-model/r/diffusion_model.R
```

**Refs:** Ho, J., Jain, A. & Abbeel, P. "Denoising diffusion probabilistic models." *NeurIPS*, 2020; Song, J., Meng, C. & Ermon, S. "Denoising diffusion implicit models (DDIM)." *ICLR*, 2021; Rombach, R. et al. "High-resolution image synthesis with latent diffusion models (Stable Diffusion)." *CVPR*, 2022; Karras, T. et al. "Elucidating the design space of diffusion-based generative models (EDM)." *NeurIPS*, 2022.

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
