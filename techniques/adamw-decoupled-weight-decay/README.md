# AdamW — Decoupled Weight Decay (Reference §47.351)

Loshchilov & Hutter (2019). Standard Adam+L2 folds the L2
penalty into the gradient BEFORE moment estimation, which
distorts the effective decay by the adaptive `√v̂`. AdamW
DECOUPLES the weight-decay step:

```
m ← β₁ m + (1 − β₁) g
v ← β₂ v + (1 − β₂) g²
m̂, v̂ ← bias-correct(m, v)
θ ← θ − η (m̂ / (√v̂ + ε) + wd · θ)
```

The `wd · θ` term is a fixed decay independent of the moment
scaling — the difference is why AdamW became the default
optimiser for transformers, ResNets, and modern LLM training.

## Files

- `python/adamw_decoupled_weight_decay.py` — ridge-like
  regression (n=200, d=30) trained 500 steps at
  `lr=0.05, wd=0.1`. AdamW test-MSE 0.116, `‖θ‖ = 3.09`;
  Adam+L2 test-MSE 0.159, `‖θ‖ = 2.82`. True `‖β‖ = 3.16` —
  AdamW is closer.
- `r/adamw_decoupled_weight_decay.R` — `torch::optim_adamw`,
  `keras::optimizer_adamw` (R); `torch.optim.AdamW`,
  `optax.adamw`, from-scratch (Python).

## When to use

- **Transformers and modern DL** — AdamW is the default in
  BERT, GPT, ViT, T5, LLaMA training recipes.
- **Fine-tuning with weight decay** — decoupled decay lets
  you tune LR and WD independently.
- **Any Adam workflow where L2 matters** — the change is one
  line of code.

## When NOT to use

- **Extremely small models** — plain SGD+momentum with
  cosine schedule can still generalise better.
- **When wd = 0** — reduces to plain Adam; no difference.
- **When you can afford full second-order (Shampoo,
  Muon)** — those exceed AdamW on some LLM benchmarks.

## Assumptions & caveats

- **wd is per-step effective** — the actual decay ratio is
  `1 − η · wd` per iteration; the ratio scales with lr.
- **Schedule interaction** — cosine / warmup schedules
  affect both lr and wd effectively; some recipes decouple
  the wd schedule too.
- **Not the same as Adam(l2=wd)** — the paper's figure 1
  shows Adam+L2 can trap the optimum inside a WD-shaped
  basin; AdamW does not.
- **Bias-correction on first steps** — early updates are
  larger; some implementations warm up wd.

## Related in this repo

- `adam-optimizer`, `rmsprop-optimizer`,
  `nesterov-accelerated-gradient`, `rectified-adam-radam`,
  `lookahead-optimizer` — first-order neighbours.
- `sam-sharpness-aware-minimization` — orthogonal
  regularisation via curvature.
- `lr-schedules`, `one-cycle-super-convergence` — LR schedule
  cousins.
- `regularization`, `ridge-lasso-elasticnet` — statistical
  regularisation background.

## Run

```
python techniques/adamw-decoupled-weight-decay/python/adamw_decoupled_weight_decay.py
Rscript techniques/adamw-decoupled-weight-decay/r/adamw_decoupled_weight_decay.R
```

**Refs:** Loshchilov, I. and Hutter, F. "Decoupled weight decay regularization." In *ICLR*, 2019.

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
