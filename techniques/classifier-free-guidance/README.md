# Classifier-Free Guidance — CFG (Reference §47.210)

Ho & Salimans (2021). Instead of using an external classifier to
steer diffusion samples toward a class, jointly train the score
model **conditional** and **unconditional** (drop the conditioning
~10 % of training). At sampling:

    ε̂ = ε_uncond + w · (ε_cond − ε_uncond)

w = 1 = standard conditional; w > 1 = **overshoot** toward
conditioning (higher fidelity, less diversity); w < 0 = **negative
prompt**.

## Files

- `python/classifier_free_guidance.py` — toy 1-D CFG demo with
  cond mode at x = 3:
  - w = 0: mean ≈ 0 (unconditional).
  - w = 1: mean = 3.6, tight around cond mode.
  - w = 3-15: mean overshoots (10 → 54), demonstrates the
    overshoot pathology at high guidance.
  - Negative w: mass pushed away from cond mode (73-87 % on the
    negative side).
- `r/classifier_free_guidance.R` — no R port; recommends
  `diffusers.StableDiffusionPipeline(guidance_scale=…)`.

## When to use

- **All text-to-image** diffusion — CFG is the default steering
  mechanism (SD, SDXL, DALL-E, Imagen).
- **Class-conditional** image gen (ImageNet).
- **Negative prompts** — describe what to AVOID.

## When NOT to use

- **When negative-prompt-free guidance suffices** (some
  training-free methods).
- **Very high w** (> 15) — introduces overshoot / saturation
  artifacts.
- **Unconditional generation** — CFG has no signal to work with.

## Assumptions & caveats

- **Dropout probability** at training (~10 %) determines the
  quality of the unconditional path.
- **Guidance scale w** typical 5-10 for images; task-specific.
- **Overshoot artifacts** at high w — colour saturation,
  detail-loss, ringing.
- **CFG-Rescale** (Lin 2024) fixes some overshoot issues.

## Related in this repo

- `diffusion-model`, `ddim-implicit-diffusion` — sampling
  neighbours.
- `latent-diffusion-ldm`, `controlnet-conditional` — full
  text-to-image pipelines using CFG.
- `rectified-flow`, `consistency-models`,
  `score-based-sde` — diffusion cousins that also use CFG.

## Run

```
python techniques/classifier-free-guidance/python/classifier_free_guidance.py
Rscript techniques/classifier-free-guidance/r/classifier_free_guidance.R
```

**Refs:** Ho, J. & Salimans, T. "Classifier-free diffusion guidance." *NeurIPS Workshop*, 2021; Lin, S. et al. "Common diffusion noise schedules and sample steps are flawed." *WACV*, 2024.

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
