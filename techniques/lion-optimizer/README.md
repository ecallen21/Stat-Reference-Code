# Lion Optimizer (Reference §47.352)

Chen et al. (2023). EvoLved Sign Momentum — discovered by
Google's Brain team via symbolic program search over 4 000
GPU-days. Update uses only the SIGN of the momentum-blended
gradient direction:

```
c ← β₁ m + (1 − β₁) g
θ ← θ − η (sign(c) + wd · θ)
m ← β₂ m + (1 − β₂) g
```

Memory: half of AdamW (no second-moment `v`). Reported to
match or exceed AdamW on ViT and LLM training with lower
FLOPs; useful default recipe uses `η = η_adamw / 10` and
`wd = 3-10 × wd_adamw`.

## Files

- `python/lion_optimizer.py` — Ridge-like regression
  (n=300, d=40). Lion and AdamW both recover `‖θ_true‖`
  to 3 decimals (MSE ≈ 0.12). Lion uses 640 bytes of state
  vs 960 bytes for AdamW (33 % memory saving, no `v` buffer).
- `r/lion_optimizer.R` — R bindings via `reticulate` +
  `lion-pytorch`; `torch.optim.Lion`, `keras.optimizers.Lion`,
  `optax.contrib.lion`, from-scratch (Python).

## When to use

- **Vision transformers, ResNets, LLM pre-training** — Lion
  paper shows small-but-consistent gains over AdamW.
- **Memory-constrained training** — no second-moment buffer.
- **When compute per step dominates memory** — Lion's step is
  cheaper (no `√v̂`).

## When NOT to use

- **Small models / small batches** — sign-based updates can
  be too coarse; use AdamW or SGD+momentum.
- **Non-DL settings** — Lion was searched for typical DL
  loss surfaces; classical convex problems don't benefit.
- **Extremely fine-tuned recipes with AdamW-specific
  schedules** — hyperparameter transfer requires re-tuning.

## Assumptions & caveats

- **LR scaling** — Lion needs ~ 10× smaller LR than AdamW
  because `sign(·)` bounds the step to `η` per dim.
- **Weight decay** — Lion effective wd is different; usually
  need 3-10× the AdamW value.
- **Gradient clipping / mixed precision** — Lion is compatible
  but often trained with `bfloat16` mixed precision.
- **β₂ near 0.99** — Lion's default is lower than AdamW's
  0.999 because the second moment is absent.

## Related in this repo

- `adamw-decoupled-weight-decay`, `adam-optimizer`,
  `rmsprop-optimizer`, `nesterov-accelerated-gradient` —
  first-order neighbours.
- `sam-sharpness-aware-minimization`,
  `stochastic-weight-averaging-swa`, `lookahead-optimizer`
  — orthogonal DL tricks.
- `evolution-strategies-openai`, `cma-es-evolution-strategy`
  — Lion was discovered by an evolutionary-style search over
  optimiser update rules.

## Run

```
python techniques/lion-optimizer/python/lion_optimizer.py
Rscript techniques/lion-optimizer/r/lion_optimizer.R
```

**Refs:** Chen, X., Liang, C., Huang, D., et al. "Symbolic discovery of optimization algorithms." arXiv:2302.06675, 2023.

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
