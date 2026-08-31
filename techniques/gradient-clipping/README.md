# Gradient Clipping (Reference Ch 30 Robustness)

Cap the magnitude of the parameter-update direction to survive
occasional huge gradients — the standard trick for training RNNs
(Pascanu, Mikolov & Bengio 2013), deep transformers, and mixed-
precision networks.

## Two flavours

**Global-norm clipping** (preserves direction):

```
if ‖g‖_2 > τ:  g ← g · τ / ‖g‖_2
```

**Value clipping** (element-wise; may distort direction):

```
g_i ← clip(g_i, −τ, τ)
```

## Why it works

- Occasional exploding gradients from long RNN horizons, near-linear
  saturation, adversarial mini-batches, mixed-precision overflow, or
  data outliers push a single SGD step far off the optimum.
- Clipping bounds the worst-case step size while keeping most steps
  untouched.

## When to use

- **RNN / LSTM / GRU** training — a default of `τ = 1.0` (norm) or
  `5.0` (value) is standard.
- **Transformer pre-training** — GPT-2/3/4-style LLMs use
  `clip_grad_norm_(τ = 1.0)`.
- **Mixed-precision (fp16 / bf16)** — combines with loss-scaling to
  prevent overflow.
- **Distributed training with occasional-slow-workers** — clipping
  guards against noisy averaged gradients.

## When NOT to use

- **Well-behaved problems** (convex, small networks) — clipping does
  nothing useful and slightly biases the update.
- **Very tight τ** — can slow convergence or bias optima; grid-search
  over {0.5, 1, 5}.
- **In place of fixing training instability** — if the loss is
  actually blowing up, clipping only masks the underlying bug.

## Files

- `python/gradient_clipping.py` — from-scratch:
  1. Single-shot demo of both clippers on a rigged outlier gradient.
  2. Linear regression under heavy-tailed gradient noise (5 % chance of
     a `scale × 100` spike). **Unclipped diverges** (β-error 1.27,
     train MSE 1.76); **norm-clip @ τ = 5** and **value-clip @ τ = 1**
     both converge (β-error 0.10 – 0.12, MSE ≈ 0.27) — 7 vs 85 clip
     events respectively.
- `r/gradient_clipping.R` — Keras / torch (R + Python); every framework
  ships both `clip_grad_norm_` and `clip_grad_value_` equivalents.

## Assumptions & caveats

- **τ vs learning rate** — reducing `τ` and reducing `lr` do similar
  things; keep both in mind when tuning.
- **Adaptive optimisers** (Adam) already renormalise per-coordinate,
  so value-clipping helps less; norm-clipping still useful.
- **Distributed sync** — clip *after* all-reducing, not before, or the
  effective threshold moves with world size.
- **Layer-wise vs global** — layer-wise clipping (LARS/LAMB) is a
  distinct method.
- **NaN gradients** — clipping does not fix NaNs; add `isfinite`
  guards separately.

## Related in this repo

- `spectral-normalization`, `jacobian-regularization` — smoothness
  priors in the *model* rather than the *optimiser*.
- `sgd-momentum`, `adam-optimizer` — the optimisers clipping wraps.
- `deep-mlp-backprop`, `state-space-models` — settings where
  exploding gradients matter most.
- `class-imbalance` — heavy-tailed losses that make clipping helpful.

## Run

```
python techniques/gradient-clipping/python/gradient_clipping.py
Rscript techniques/gradient-clipping/r/gradient_clipping.R
```

**Refs:** Pascanu, R., Mikolov, T. & Bengio, Y. "On the difficulty of training recurrent neural networks." *ICML*, 2013; You, Y. et al. "Large batch optimisation for deep learning: training BERT in 76 minutes (LAMB)." *ICLR*, 2020.

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
