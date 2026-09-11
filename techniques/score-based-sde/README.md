# Score-Based Generative Modeling via SDEs (Reference §47.212)

Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole (2021, ICLR).
Unifies DDPM and score-matching under a continuous-time SDE
framework:

    dx = f(x, t) dt + g(t) dW               (forward SDE)
    dx = [f(x, t) − g(t)² · ∇log p_t(x)] dt + g(t) dW̄  (reverse SDE)

Learn a score network s_θ(x, t) ≈ ∇_x log p_t(x). Sample by
integrating the **reverse SDE** (Euler-Maruyama) or the equivalent
**deterministic probability-flow ODE** (adaptive ODE solver).

## Files

- `python/score_based_sde.py` — Euler-Maruyama reverse-SDE
  sampling with an ORACLE score for a 3-mode 2-D Gaussian mixture:
  - Truth: means [-2, 0], [2, 0], [0, 2].
  - Recovered cluster centres via reverse-SDE (200 steps, oracle
    score): [-1.96, -0.02], [2.01, 0.03], [-0.04, 1.95] — mode
    counts 179/165/156 (balanced).
- `r/score_based_sde.R` — no R port; recommends
  `yang-song/score_sde`, `diffusers.ScoreSdeVeScheduler`.

## When to use

- **Framework-level thinking** about diffusion — SDEs unify DDPM,
  score-matching, VE / VP formulations.
- **When you need adaptive ODE solvers** — PF-ODE lets you use
  Dopri5 / DPM-Solver for high-order sampling.
- **Continuous-time analysis** of generative models.

## When NOT to use

- **When a specific discretisation works well** — DDIM / DPM-
  Solver already implement the ODE.
- **Very sharp / bounded distributions** — VE-SDE's exploding
  noise struggles.
- **When you don't need SDE-level flexibility** — simpler
  schedulers suffice for standard image gen.

## Assumptions & caveats

- **VE (variance-exploding)** vs **VP (variance-preserving)** SDE
  — different noise schedules, both work.
- **Predictor-Corrector** (PC) sampling mixes SDE steps with MCMC
  Langevin corrections.
- **Probability-flow ODE** = deterministic reformulation with
  the same marginals as the SDE.
- **Score-matching loss** at each t requires denoising-score-
  matching estimator (Vincent 2011).

## Related in this repo

- `diffusion-model`, `ddim-implicit-diffusion`,
  `classifier-free-guidance` — diffusion-family cousins.
- `neural-ode`, `state-space-models` — continuous-time
  neighbours.
- `latent-diffusion-ldm`, `rectified-flow`,
  `consistency-models` — subsequent diffusion advances.

## Run

```
python techniques/score-based-sde/python/score_based_sde.py
Rscript techniques/score-based-sde/r/score_based_sde.R
```

**Refs:** Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S. & Poole, B. "Score-based generative modeling through stochastic differential equations." *ICLR*, 2021; Vincent, P. "A connection between score matching and denoising autoencoders." *Neural Computation* 23(7), 2011.

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
