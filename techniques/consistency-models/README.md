# Consistency Models (Reference §47.215)

Song, Dhariwal, Chen & Sutskever (2023, ICML). Train a model
f_θ(x_t, t) that maps ANY point on a probability-flow ODE
trajectory back to the trajectory's ORIGIN x_0:

    f_θ(x_t, t) = x_0     (self-consistency along the ODE)

Loss: L_CM = d(f_θ(x_{t+1}, t+1), f_θ_prev(x_t, t))
     (distillation of a pretrained diffusion model, or
      consistency-training from scratch)

Enables **one-step generation** while retaining most of the
quality of a multi-step diffusion sampler.

## Files

- `python/consistency_models.py` — toy CM training on a 3-mode
  Gaussian mixture in 2-D:
  - **1-step generation** with the trained linear consistency
    model: mean dist to nearest mode = 1.72.
  - Samples per mode: 117 / 145 / 238 (all 3 modes represented).
- `r/consistency_models.R` — no R port; recommends
  `openai/consistency_models`, `diffusers.CMStochasticIterativeScheduler`,
  latent-consistency-model variants.

## When to use

- **1-4 step text-to-image** generation — LCM, LCM-LoRA on SD.
- **Real-time generative** applications (interactive tools).
- **Distilling** a slow diffusion model into a faster one.

## When NOT to use

- **When the diffusion base is small** — training a CM adds
  complexity for little payoff.
- **When maximum sample quality matters** and 20-50 step
  diffusion is affordable.
- **Highly multimodal, high-resolution targets** — 1-step CM can
  miss modes.

## Assumptions & caveats

- **Distillation vs training from scratch** — distillation is
  simpler; from-scratch (CT) more expensive but no teacher.
- **EMA teacher** — training uses an exponential-moving-average
  of the student as the target model.
- **Multi-step CM sampling** — 2-4 steps can beat 1-step
  meaningfully.
- **Improved variants**: iCM (Song 2024), Latent CM, Multistep
  CM.

## Related in this repo

- `diffusion-model`, `ddim-implicit-diffusion`,
  `score-based-sde`, `rectified-flow` — diffusion-family
  neighbours.
- `latent-diffusion-ldm` — LCM commonly applied here.
- `classifier-free-guidance`, `controlnet-conditional` —
  compatible steering methods.

## Run

```
python techniques/consistency-models/python/consistency_models.py
Rscript techniques/consistency-models/r/consistency_models.R
```

**Refs:** Song, Y., Dhariwal, P., Chen, M. & Sutskever, I. "Consistency models." *ICML*, 2023; Luo, S. et al. "Latent Consistency Models: Synthesizing high-resolution images with few-step inference." *arXiv:2310.04378*, 2023.

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
