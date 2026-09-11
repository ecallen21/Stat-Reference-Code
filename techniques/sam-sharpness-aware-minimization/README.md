# Sharpness-Aware Minimization — SAM (Reference §47.163)

Foret et al. (2021). Instead of minimising the loss at θ, SAM
minimises the loss in a **neighbourhood** around θ:

    min_θ max_{‖ε‖ ≤ ρ} L(θ + ε).

Two-step update:

    1. ε* = ρ · ∇L(θ) / ‖∇L(θ)‖         (ascent)
    2. θ ← θ − lr · ∇L(θ + ε*)          (descent)

Doubles compute per step, but biases training toward **flat
minima** that generalise better — especially with limited data or
label noise.

## Files

- `python/sam_sharpness_aware_minimization.py` — 1-hidden-layer
  MLP (h = 16) on `make_moons` with 10 % label noise:
  - SGD  test acc = **0.757**, sharpness = 0.0100
  - **SAM  test acc = 0.771**, sharpness = **0.0028** — 3.6× flatter
    basin, +1.4 pt test accuracy.
- `r/sam_sharpness_aware_minimization.R` — no native R port;
  recommends `sam-optimizer` (davda54/sam), `torch_optimizer.SAM`.

## When to use

- **Vision / NLP models with limited data** where flat-minima
  regularisation helps.
- **Noisy labels** — SAM smooths over noisy loss landscape.
- **Domain-generalisation** benchmarks — SAM consistently helps.

## When NOT to use

- **Tight compute budget** — SAM doubles per-step time.
- **Convex problems** — no flatness distinction; plain SGD is
  fine.
- **Very small nets** — the benefit shrinks with capacity.

## Assumptions & caveats

- **ρ (neighbourhood radius)** typically 0.05-0.2; too large
  hurts, too small = no effect.
- **First gradient is stochastic** — noisy ascent direction, but
  the paper shows this is acceptable.
- **ASAM (adaptive SAM)** and **ESAM (efficient SAM)** are
  variants for cost / scale-invariance.
- **Momentum interactions** — apply SAM's descent step to
  momentum-updated fast params.

## Related in this repo

- `stochastic-weight-averaging-swa` — sibling flat-minima
  method.
- `lookahead-optimizer` — cousin two-timescale training method.
- `label-smoothing`, `mixup`, `cutmix` — generalisation
  regularisers.
- `spectral-normalization`, `jacobian-regularization` — related
  loss-landscape shaping techniques.

## Run

```
python techniques/sam-sharpness-aware-minimization/python/sam_sharpness_aware_minimization.py
Rscript techniques/sam-sharpness-aware-minimization/r/sam_sharpness_aware_minimization.R
```

**Refs:** Foret, P., Kleiner, A., Mobahi, H. & Neyshabur, B. "Sharpness-aware minimization for efficiently improving generalization." *ICLR*, 2021; Kwon, J. et al. "ASAM: Adaptive sharpness-aware minimization." *ICML*, 2021; Du, J. et al. "Efficient sharpness-aware minimization for improved training of neural networks." *ICLR*, 2022.

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
