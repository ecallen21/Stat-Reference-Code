# Rectified Flow (Reference §47.214)

Liu, Gong & Liu (2023). Reformulates diffusion as a **straight-
line flow** between prior x_0 ~ N(0, I) and data x_1 ~ p:

    x_t = (1 − t) x_0 + t x_1,  t ∈ [0, 1]
    dx/dt = x_1 − x_0

Learn a velocity net v_θ(x, t) ≈ E[x_1 − x_0 | x_t = x].
**Reflow** iterates: use the sampler's trajectories as new
(x_0, x_1) couplings, yielding straighter paths and 1-step
generation.

## Files

- `python/rectified_flow.py` — from-scratch linear velocity net
  on a source-N(0, I) → target-2-mode-mixture flow:
  - **1 Euler step**: mean dist to nearest mode = 1.61.
  - 4 steps: 1.58.
  - 20 steps: 1.65.
  - Similar quality at 1 vs 20 steps because rectified paths are
    close to straight.
- `r/rectified_flow.R` — no R port; recommends
  `gnobitab/RectifiedFlow`, `diffusers.FlowMatchEulerDiscreteScheduler`.

## When to use

- **Ultra-fast diffusion sampling** — SD3, Flux use rectified
  flow at inference.
- **When trajectories are near-straight** — training and sampling
  are simpler than diffusion SDEs.
- **Coupling different distributions** — general flow between two
  arbitrary priors.

## When NOT to use

- **When maximum sample quality is required** and you can afford
  many steps — full diffusion sometimes wins.
- **Highly multimodal targets from unimodal prior** — 1-step
  averaging may miss modes.
- **When existing DDPM checkpoints exist** — retraining as
  rectified flow is a big investment.

## Assumptions & caveats

- **Reflow iterations** — each round straightens the path
  further; 1-2 rounds typical.
- **Coupling** — Optimal Transport couplings (OT-CFM, Tong 2023)
  find best pairing before training.
- **Timestep parametrisation** — uniform t sampling standard;
  logit-normal (SD3) helps at boundaries.
- **Same score / flow duality** — rectified flow's velocity is
  related to diffusion's score by a linear transform.

## Related in this repo

- `diffusion-model`, `ddim-implicit-diffusion`,
  `score-based-sde`, `consistency-models` — diffusion-family
  neighbours.
- `latent-diffusion-ldm`, `classifier-free-guidance`,
  `controlnet-conditional` — t2i pipeline neighbours.
- `neural-ode` — deterministic ODE flow cousin.

## Run

```
python techniques/rectified-flow/python/rectified_flow.py
Rscript techniques/rectified-flow/r/rectified_flow.R
```

**Refs:** Liu, X., Gong, C. & Liu, Q. "Flow straight and fast: Learning to generate and transfer data with rectified flow." *ICLR*, 2023; Esser, P. et al. "Scaling rectified flow transformers for high-resolution image synthesis (SD3)." *ICML*, 2024; Tong, A. et al. "Improving and generalizing flow-based generative models with minibatch optimal transport." *ICML*, 2024.

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
