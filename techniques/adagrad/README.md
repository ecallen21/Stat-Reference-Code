# Adagrad (Reference §47.353)

Duchi, Hazan & Singer (2011). Per-parameter adaptive learning
rate that shrinks proportionally to the historical
squared gradient:

```
G_t = G_{t-1} + g_t²
θ = θ − η / (√G_t + ε) · g_t
```

Sparse features get **large** updates (small G), dense ones
get **small** updates. Regret bound `O(√T)` for online convex
optimisation, tighter for sparse gradients (Duchi's data-
dependent bound).

## Files

- `python/adagrad.py` — SGD vs Adagrad on a sparse regression
  problem (n=400, d=50, only 10% non-zero features, 10 truly
  nonzero coefficients). Adagrad reaches MSE 0.045 with tight
  coefficient recovery (nonzero-err 0.22); SGD stalls at MSE
  0.084 with much noisier coefficients (nonzero-err 1.06).
- `r/adagrad.R` — `torch::optim_adagrad`,
  `keras::optimizer_adagrad` (R); `torch.optim.Adagrad`,
  `optax.adagrad`, from-scratch (Python).

## When to use

- **Sparse features** — NLP, click-through-rate, high-dim
  linear models where most gradients are zero.
- **Convex online learning** — sub-linear regret guarantee.
- **Rare-event / rare-feature scenarios** — the automatic
  per-parameter step scaling handles very unbalanced updates.

## When NOT to use

- **Deep neural networks** — G grows monotonically, learning
  rate decays to zero prematurely; use RMSProp / Adam / AdamW
  instead.
- **Non-convex, long-training runs** — same monotonicity
  problem stalls learning.
- **When you can afford Adam** — Adam's exponential moving
  average removes the decay-to-zero problem.

## Assumptions & caveats

- **Monotone learning rate** — the fatal flaw for deep nets;
  patched by RMSProp / Adam / AdaDelta.
- **Initial LR** — Adagrad tolerates larger LR (0.1-1.0)
  because of automatic per-feature scaling.
- **eps in denominator** — 1e-8 default; larger values
  stabilise very early updates.
- **Sparse implementations** — `torch.optim.Adagrad` accepts
  sparse tensors for CTR / embedding tables.
- **Composite regularisation** — Duchi's paper includes a
  proximal / composite version for L1 regularisation.

## Related in this repo

- `adam-optimizer`, `rmsprop-optimizer`,
  `adamw-decoupled-weight-decay`, `nadam-optimizer` —
  adaptive-optimiser cousins that fix Adagrad's decay.
- `lion-optimizer`, `nesterov-accelerated-gradient`,
  `sag-saga-variance-reduction` — first-order alternatives.
- `online-learning-sgd`, `stochastic-gradient-mcmc` — SGD
  variants.

## Run

```
python techniques/adagrad/python/adagrad.py
Rscript techniques/adagrad/r/adagrad.R
```

**Refs:** Duchi, J., Hazan, E. and Singer, Y. "Adaptive subgradient methods for online learning and stochastic optimization." *J. Mach. Learn. Res.*, 12: 2121-2159, 2011.

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
