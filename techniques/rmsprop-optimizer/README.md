# RMSprop Optimizer (Reference §47.106)

Tieleman & Hinton (2012, Coursera lecture 6e). Divides each
coordinate's gradient by a running RMS of its squared gradients:

    v_t = β v_{t−1} + (1 − β) g_t²
    θ_t = θ_{t−1} − η g_t / (√v_t + ε).

Adaptive per-coordinate step; predecessor of **Adam** (which adds
momentum on g_t itself). Popular for RNNs and noisy-gradient
regimes.

## Files

- `python/rmsprop_optimizer.py` — from-scratch SGD / RMSprop /
  Adam on an ill-conditioned quadratic `f(x) = 0.5 xᵀ diag(1, 100) x`.
  Demo:
  - SGD lr=0.02: ‖θ‖ = 1.02 after 200 iters (stuck on stiff dim)
  - RMSprop lr=0.5: ‖θ‖ = 0.35 (adaptive per-coord rescaling)
  - Adam lr=0.5: ‖θ‖ = 2.9 × 10⁻⁴ (adds momentum on top).
  Noisy-gradient case shows the opposite: RMSprop beats Adam
  because Adam's momentum amplifies noise.
- `r/rmsprop_optimizer.R` — `torch::optim_rmsprop` (R
  torch bindings); `torch.optim.RMSprop`, `keras.optimizers.RMSprop`
  (Python).

## When to use

- **Recurrent nets** — Hinton's original recommendation.
- **Non-stationary objectives** where per-coord scale changes.
- **Noisy stochastic gradients** — Adam's momentum can be too
  aggressive.
- **When a plain SGD baseline seems slow** — RMSprop often works
  out-of-the-box.

## When NOT to use

- **Deterministic convex smooth** — Nesterov / L-BFGS is optimal.
- **When Adam clearly wins** empirically — no reason to switch back.
- **Convex sparse regression** — coord descent / proximal methods
  dominate.

## Assumptions & caveats

- **β (running-average momentum)** default 0.9; larger = smoother.
- **ε** default 1e-8; too small → division-by-zero blowup at start.
- **Learning-rate warmup / decay** still often helps.
- **Weight decay** ≠ L2 penalty for adaptive methods — decouple
  (AdamW / RMSprop-W).

## Related in this repo

- `adam-optimizer`, `nesterov-accelerated-gradient`,
  `online-learning-sgd`, `lr-schedules`, `gradient-clipping` —
  optimisation toolbox.
- `deep-ensembles`, `mc-dropout`, `bayesian-neural-network` —
  models trained by these optimisers.
- `stochastic-gradient-mcmc`, `variational-inference` — Bayesian
  training routines.

## Run

```
python techniques/rmsprop-optimizer/python/rmsprop_optimizer.py
Rscript techniques/rmsprop-optimizer/r/rmsprop_optimizer.R
```

**Refs:** Tieleman, T. & Hinton, G. "Lecture 6e: RMSprop – Divide the gradient by a running average of its recent magnitude." *Coursera: Neural Networks for Machine Learning*, 2012; Kingma, D.P. & Ba, J. "Adam: A method for stochastic optimization." *ICLR*, 2015.

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
