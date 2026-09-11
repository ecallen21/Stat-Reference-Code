# DDIM — Denoising Diffusion Implicit Models (Reference §47.209)

Song, Meng & Ermon (2021, ICLR). Reformulates the DDPM reverse
process as a **non-Markovian, deterministic ODE** that can be
integrated with far fewer steps:

    x_{t−1} = √α_{t−1} · x̂_0(x_t, ε̂)  +  √(1 − α_{t−1}) · ε̂

Same trained score model as DDPM, but 10-50 steps instead of
1000. Deterministic sampling enables meaningful latent-space
interpolation.

## Files

- `python/ddim_implicit_diffusion.py` — DDPM vs DDIM sampling on a
  2-mode Gaussian mixture using an oracle score:
  - **DDPM (100 steps)**: avg reconstruction distance = 0.0000.
  - **DDIM (10 steps)**: avg distance = 0.0000 — 10× fewer steps
    at identical quality with the oracle score.
- `r/ddim_implicit_diffusion.R` — no R port; recommends
  `diffusers.DDIMScheduler`, `ermongroup/ddim`.

## When to use

- **Any DDPM-trained model** at inference time — DDIM is a
  drop-in scheduler.
- **Latent-space interpolation** — deterministic paths give
  meaningful traversals.
- **Speed-critical serving** — 20-50 DDIM steps ≈ 1000 DDPM steps
  on quality.

## When NOT to use

- **When maximum sample diversity is required** — DDIM's
  deterministic paths are less diverse than stochastic DDPM.
- **When quality at 1-4 steps matters** — use consistency /
  rectified-flow models instead.
- **Score model trained specifically for one scheduler** — swap
  usually works but retune the noise schedule.

## Assumptions & caveats

- **Eta parameter** ∈ [0, 1]: eta = 0 = deterministic DDIM,
  eta = 1 = DDPM.
- **Number of sampling steps** — 20-50 typical for image gen; 10
  is often noticeably worse.
- **Noise schedule** shared with DDPM — no retraining needed.
- **Related samplers**: DPM-Solver, Heun, Euler-Ancestral all
  compete with DDIM at 10-30 steps.

## Related in this repo

- `diffusion-model` — DDPM training pipeline DDIM samples from.
- `score-based-sde`, `rectified-flow`, `consistency-models` —
  diffusion-family cousins.
- `classifier-free-guidance`, `latent-diffusion-ldm`,
  `controlnet-conditional` — DDIM's typical partners in text-to-
  image pipelines.

## Run

```
python techniques/ddim-implicit-diffusion/python/ddim_implicit_diffusion.py
Rscript techniques/ddim-implicit-diffusion/r/ddim_implicit_diffusion.R
```

**Refs:** Song, J., Meng, C. & Ermon, S. "Denoising diffusion implicit models." *ICLR*, 2021; Ho, J., Jain, A. & Abbeel, P. "Denoising diffusion probabilistic models." *NeurIPS*, 2020.

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
