# Lookahead Optimizer (Reference §47.164)

Zhang, Lucas, Ba & Hinton (2019). Keeps **two sets of weights**:

- **φ** ("slow" weights), and
- **θ** ("fast" weights, the SGD state).

    For k inner steps: θ ← θ − lr · ∇L(θ).
    Then:              φ ← φ + α · (θ − φ)
                       θ ← φ                    (reset fast to slow)

α ∈ (0, 1). Wraps any base optimiser; robust to lr choice, reduces
variance, and improves generalisation with negligible compute
overhead.

## Files

- `python/lookahead_optimizer.py` — SGD-momentum vs Lookahead
  (wraps the same SGD) on a 30-D, 800-sample classification. Sweeps
  learning rate:
  - lr = 0.01: SGD 0.737 vs LA 0.720.
  - lr = 1.00 (extreme): **SGD 0.697 vs LA 0.707** — Lookahead
    more robust at the tails of the lr range.
- `r/lookahead_optimizer.R` — no native R port; recommends
  `pytorch_optimizer.Lookahead` and `RangerOptimizer` (RAdam +
  Lookahead).

## When to use

- **Any deep-net training** — Lookahead is a nearly-free wrapper.
- **When lr tuning is expensive** — Lookahead widens the good-lr
  window.
- **Combined with RAdam** as Ranger — a popular robust default.

## When NOT to use

- **When base optimiser is already well-tuned** and stable, gain
  is marginal.
- **Ultra-tight step budget** — the k-step slow-update rhythm
  needs enough steps to matter.
- **Single-machine research** where reproducibility of a specific
  optimiser matters (adds a variance-reduction confound).

## Assumptions & caveats

- **α (slow-weight step size)** — 0.5 is a robust default.
- **k (inner steps)** — 5-10 typical; larger = smoother, slower to
  respond.
- **Reset vs no-reset**: paper resets θ ← φ every k steps; some
  impls don't reset.
- **Interaction with schedulers**: apply the schedule to the fast
  optimiser's lr.

## Related in this repo

- `adam-optimizer`, `lr-schedules`, `gradient-clipping` — training
  optimiser toolbox.
- `rectified-adam-radam` — the classic Lookahead pair (Ranger).
- `stochastic-weight-averaging-swa`,
  `sam-sharpness-aware-minimization` — related weight-averaging
  ideas.

## Run

```
python techniques/lookahead-optimizer/python/lookahead_optimizer.py
Rscript techniques/lookahead-optimizer/r/lookahead_optimizer.R
```

**Refs:** Zhang, M. R., Lucas, J., Ba, J. & Hinton, G. E. "Lookahead optimizer: k steps forward, 1 step back." *NeurIPS*, 2019; Wright, L. "Ranger: RAdam + Lookahead optimizer for deep learning." *GitHub*, 2019.

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
