# Learning-Rate Schedules (Reference §27.x extra)

The learning rate is the single most important hyperparameter of a neural-net
training run. A schedule adjusts it over time.

## Five workhorse schedules

| Schedule | Formula | Best for |
|---|---|---|
| **Constant** | `lr(t) = lr_0` | quick experiments, warm starts |
| **Step decay** | `lr_0 · γ^⌊t / step⌋` | classical CNN training (ResNet on ImageNet) |
| **Cosine** | `lr_min + (lr_0 − lr_min) · (1 + cos(πt / T)) / 2` | modern default in most transformers |
| **Warmup + cosine** | linear ramp for first `w` steps, then cosine to 0 | transformers, LLMs, any post-norm arch |
| **One-cycle** (Smith 2018) | linear up to `lr_max` then cosine down | fastai-style rapid CNN training; also very good for tabular tasks |

## Related schedules

- **CosineAnnealingWarmRestarts** — periodic cosine cycles; SGDR (Loshchilov-Hutter).
- **ReduceLROnPlateau** — drop LR when validation loss stops improving.
- **Inverse-sqrt** — Noam schedule (`lr ∝ 1 / √t`); original transformer paper.
- **Polynomial decay** — `lr_0 · (1 − t / T)^p`; NLP fine-tuning.
- **Cyclical LR** (Smith 2015) — triangular cycles across an LR band.

## Why warmup?

Adam's early updates use small `v̂_t`, so the effective step size is large;
starting at `lr_0` from step 1 often diverges post-norm transformers.
Warmup ramps `lr` linearly for the first 1–5% of steps to let `v̂_t` stabilise.

## When to use

- **Transformer / LLM fine-tuning** — warmup + cosine or warmup + linear-to-0.
- **CNN classifier training** — step decay (classical) or one-cycle (fastai / rapid).
- **Small-dataset training** — one-cycle often beats everything.
- **Very long training runs** — cosine restarts (SGDR).
- **Unknown-length training** — ReduceLROnPlateau (needs a validation loss).

## Files

- `python/lr_schedules.py` — from-scratch numpy implementations of all five. Demo over T=100 steps at `lr_0 = 0.1`:
  - **Constant** flat at 0.1.
  - **Step (30, 0.1)** drops from 0.1 → 0.01 → 0.001 → 0.0001 at steps 30, 60, 90.
  - **Cosine** smooth 0.1 → 0.05 (step 50) → 0.0.
  - **Warmup+cosine** ramp to 0.10 at step 9, then decay to 0.0.
  - **One-cycle** ramp to 0.10 at step 50, then cosine to 0.0.
- `r/lr_schedules.R` — `torch::lr_scheduler_*`, `keras3::callback_learning_rate_scheduler`; Python `torch.optim.lr_scheduler.*`, `transformers.get_scheduler`, `optax` (JAX).

## Assumptions & caveats

- **Global vs per-parameter-group LR** — LLM fine-tuning often uses different LRs for different layers (LR decay per depth, embedding vs head).
- **LR peak scales with batch size** — linear scaling rule (Goyal 2017); use with warmup.
- **Cosine "to zero" is a common mistake if training may resume** — leave a floor `lr_min`.
- **One-cycle mid-cycle validation is misleading** — the model is not converged; only end-of-cycle val loss is meaningful.
- **AdamW + warmup + cosine** is the near-universal transformer recipe; deviate with reason.
- **Schedules interact with weight-decay schedules** — some recipes (T5) decouple decay-decay from LR-decay.

## Related in this repo

- `adam-optimizer` — the optimiser that these schedules control.
- `deep-mlp-backprop`, `convolutional-nn`, `transformer-encoder` — architectures where these are used.
- `bayesian-optimization` — an alternative for choosing the schedule / hyperparameters.
- `nested-cv` — the honest way to tune schedule hyperparameters.

## Run

```
python techniques/lr-schedules/python/lr_schedules.py
Rscript techniques/lr-schedules/r/lr_schedules.R
```

**Refs:** Loshchilov, I. & Hutter, F. "SGDR: stochastic gradient descent with warm restarts." *ICLR*, 2017; Smith, L.N. "A disciplined approach to neural network hyper-parameters: part 1—learning rate, batch size, momentum, and weight decay." *arXiv:1803.09820*, 2018; Goyal, P. et al. "Accurate, large minibatch SGD: training ImageNet in 1 hour." *arXiv:1706.02677*, 2017.

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
