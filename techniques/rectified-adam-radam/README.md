# Rectified Adam — RAdam (Reference §47.165)

Liu et al. (2020). Adam's adaptive learning rate has **high
variance** in the first few steps; a common fix is a linear warm-
up. RAdam derives a **closed-form rectification** term that turns
Adam into plain SGD (no adaptive lr) while its variance estimate is
unreliable, then smoothly switches to Adam once enough gradient
history has accumulated.

    ρ∞     = 2 / (1 − β₂) − 1
    ρ_t    = ρ∞ − 2 t β₂^t / (1 − β₂^t)
    if ρ_t > 4:  r_t = √((ρ_t − 4)(ρ_t − 2) ρ∞ /
                         ((ρ∞ − 4)(ρ∞ − 2) ρ_t))
                 Adam step · r_t
    else:        plain SGD step (no adaptive lr).

## Files

- `python/rectified_adam_radam.py` — Adam vs RAdam on a noisy
  ill-conditioned quadratic bowl in 5-D:
  - Long-run: both converge; Adam slightly better on this convex
    problem.
  - **Early iters (noise = 3.0, first 20 steps)**: Adam loss
    trajectory [63, 54, 46, 40, 34] vs **RAdam [48, 15, 15, 15, 15]** —
    RAdam converges much faster because it doesn't fire the noisy
    adaptive lr until ρ_t > 4.
- `r/rectified_adam_radam.R` — no native R port; recommends
  `pytorch_optimizer.RAdam`, `torch.optim.RAdam`.

## When to use

- **Any Adam workload** — RAdam is a drop-in replacement.
- **When you'd otherwise use linear-lr warm-up** — RAdam
  automates it.
- **Small-batch training** where adaptive-lr variance is highest.

## When NOT to use

- **When you're already using SAM / Ranger / another
  well-tuned pipeline** — extra tuning cost may not be worth it.
- **Fixed lr / SGD-only workflows** — nothing to rectify.
- **Very stable, well-conditioned problems** — Adam / RAdam
  differ negligibly.

## Assumptions & caveats

- **β₂ = 0.999** is the default; smaller values (0.99) make the
  SGD-warmup phase longer.
- **The rectification factor r_t → 1 as t → ∞**, so RAdam
  asymptotically matches Adam.
- **Combined with Lookahead** as "Ranger" — popular robust
  default.
- **Weight decay** is decoupled in AdamW; RAdamW variant exists.

## Related in this repo

- `adam-optimizer`, `lr-schedules`, `gradient-clipping` —
  optimiser toolbox.
- `lookahead-optimizer` — Ranger pair.
- `one-cycle-super-convergence` — alternative lr-schedule
  strategy.

## Run

```
python techniques/rectified-adam-radam/python/rectified_adam_radam.py
Rscript techniques/rectified-adam-radam/r/rectified_adam_radam.R
```

**Refs:** Liu, L. et al. "On the variance of the adaptive learning rate and beyond." *ICLR*, 2020; Kingma, D. P. & Ba, J. "Adam: A method for stochastic optimization." *ICLR*, 2015.

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
