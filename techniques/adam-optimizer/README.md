# Adam and Friends (Reference §27.11)

Five workhorse stochastic-gradient optimisers.

## Update rules

| Optimiser | Update | Notes |
|---|---|---|
| **SGD** | `w ← w − lr · g` | classical; needs LR tuning |
| **Momentum SGD** | `m ← β₁ m + g;  w ← w − lr · m` | accelerates flat directions |
| **Nesterov** | evaluate g at lookahead position `w − lr · β₁ m` | slightly tighter convergence than Momentum |
| **RMSProp** | `v ← β₂ v + (1 − β₂) g²;  w ← w − lr · g / (√v + ε)` | per-parameter step size |
| **Adam** (Kingma-Ba 2015) | `m, v` accumulators + **bias-corrected** `m̂, v̂`; `w ← w − lr · m̂ / (√v̂ + ε)` | momentum + RMSProp with bias correction |
| **AdamW** (Loshchilov-Hutter 2019) | Adam with **decoupled** weight decay `w ← w − lr · wd · w` | proper L2 regularisation for Adam |

## Adam details

```
m_t = β₁ m_{t−1} + (1 − β₁) g_t          first-moment (momentum)
v_t = β₂ v_{t−1} + (1 − β₂) g_t²         second-moment (RMSProp)
m̂_t = m_t / (1 − β₁ᵗ),  v̂_t = v_t / (1 − β₂ᵗ)   bias correction
w_t = w_{t−1} − lr · m̂_t / (√v̂_t + ε)
```

Default hyperparameters: `lr = 1e-3, β₁ = 0.9, β₂ = 0.999, ε = 1e-8`.

## When to use

- **Adam / AdamW** — the default for everything neural (LLMs, ViTs, diffusion, ResNets).
- **SGD + Momentum** — classical CNN training (some ResNets prefer it with careful LR schedule).
- **RMSProp** — RNN / older LSTMs; largely superseded by Adam.
- **LBFGS** — small deterministic problems (10²–10⁴ params).
- **Lion / LAMB / Adafactor** — LLM-scale training with memory constraints.

## Files

- `python/adam_optimizer.py` — from-scratch numpy implementations of SGD, Momentum, RMSProp, Adam, AdamW. Demos:
  - **Sphere** `f(w) = ||w||²` from `w₀ = [3, 4]`, 200 iterations → all five reach ||w|| < 1e-2.
  - **Rosenbrock** (`f(x, y) = (1 − x)² + 100 (y − x²)²`) from `[-1.2, 1.0]`, 5000 iterations → Momentum, Adam, AdamW reach `(1, 1)` exactly; SGD stops at `(0.94, 0.88)`; RMSProp at `(0.98, 0.97)`.
- `r/adam_optimizer.R` — `torch::optim_sgd / rmsprop / adam / adamw / lbfgs`; Python `torch.optim.*`, `optax`, `keras.optimizers`.

## Assumptions & caveats

- **Adam's L2 in the loss ≠ true weight decay** — Loshchilov-Hutter showed Adam's momentum couples with the L2 penalty in a bad way; use AdamW for anything past a warmup.
- **Adam-generalisation gap** — Adam-trained models sometimes generalise worse than SGD+Momentum-trained ones (Wilson et al. 2017). Debate ongoing; task-dependent.
- **Learning-rate scheduling matters more than the optimiser** — warmup + cosine decay + a floor is the standard modern recipe.
- **β₂ close to 1** slows adaptation but is essential for stability at start; `1 − β₂ᵗ` bias correction handles the initial-transient burn-in.
- **Gradient clipping** (by norm) pairs with any optimiser; standard for RNNs and transformers.
- **Second-order methods** (Newton, natural gradient, K-FAC) rarely justify their cost at scale; Adam's diagonal preconditioning is usually enough.

## Related in this repo

- `deep-mlp-backprop`, `convolutional-nn`, `transformer-encoder` — architectures trained with these.
- `dropout-batchnorm` — regularisation counterpart.
- `online-learning-sgd` — the streaming special case with per-example updates.
- `bayesian-optimization`, `laplace-approximation` — second-order / hyper-parameter counterparts.

## Run

```
python techniques/adam-optimizer/python/adam_optimizer.py
Rscript techniques/adam-optimizer/r/adam_optimizer.R
```

**Refs:** Kingma, D.P. & Ba, J. "Adam: a method for stochastic optimization." *ICLR*, 2015; Loshchilov, I. & Hutter, F. "Decoupled weight decay regularization." *ICLR*, 2019; Wilson, A.C. et al. "The marginal value of adaptive gradient methods in machine learning." *NeurIPS*, 2017.

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
