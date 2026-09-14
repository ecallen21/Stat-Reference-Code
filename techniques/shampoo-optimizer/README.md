# Shampoo (Reference §47.357)

Gupta, Koren & Singer (2018). Preconditioned Stochastic Tensor
Optimization — a Kronecker-factored approximation to full-
matrix Adagrad. For a `(r × c)` weight matrix `W`, maintain
two preconditioners:

```
L ← β L + (1 − β) g gᵀ         (r × r)
R ← β R + (1 − β) gᵀ g          (c × c)
W ← W − η L^(−1/4) g R^(−1/4)
```

Storage `O(r² + c²)` instead of full-Adagrad's `O((rc)²)`.
The fourth-root exponent balances curvature information along
both matrix modes. Recent variants (distributed Shampoo,
Muon) power PaLM-2 and other frontier LLMs.

## Files

- `python/shampoo_optimizer.py` — Matrix regression
  (20×15 weight, n=200). Shampoo and SGD both converge to
  `‖W‖ ≈ 8.71` (truth 8.73) with MSE 0.019. The point of
  interest: Shampoo state 5 000 B (L + R) vs
  full-Adagrad-equivalent Hessian 720 000 B — 99% smaller.
- `r/shampoo_optimizer.R` — reticulate to Google's
  `scalable_shampoo` (R); `optax.scale_by_shampoo`,
  `pytorch-optimizer.Shampoo`, from-scratch (Python).

## When to use

- **Large transformer / LLM training** — Distributed Shampoo
  is state-of-the-art on some pretraining tasks.
- **When curvature information helps** — problems where
  AdamW's diagonal preconditioner leaves gains on the table.
- **When full-Adagrad's memory is prohibitive** — Kronecker
  form is a sweet spot.

## When NOT to use

- **Small models / small matrices** — the overhead of
  matrix inverse roots doesn't pay back.
- **1-D parameters** — Shampoo degenerates to Adagrad on
  vectors.
- **Online training with rapid distribution shift** — the
  preconditioner cache stales.

## Assumptions & caveats

- **Matrix-power computation** — `A^(−1/4)` via eigendecomp
  or Coupled-Newton iteration; refreshed periodically
  (typically every 100-500 steps).
- **Numerical conditioning** — add `ε I` before inverting;
  Coupled-Newton in bfloat16 is delicate.
- **Grafting** — production Shampoo often "grafts" step
  norms from AdamW to inherit its stability while using
  Shampoo's direction.
- **Distributed variant** — Anil et al 2020 add sharding
  and pipelining for very large models.

## Related in this repo

- `adamw-decoupled-weight-decay`, `adagrad`,
  `adam-optimizer`, `lion-optimizer` — first-order
  neighbours.
- `adafactor` — matrix-factorised second moment.
- `lbfgs-quasi-newton`, `conjugate-gradient-cg`,
  `proximal-newton` — deterministic second-order cousins.
- `zero-redundancy-optimizer`, `fsdp-fully-sharded-data-parallel`,
  `pipeline-parallelism` — distributed training partners.

## Run

```
python techniques/shampoo-optimizer/python/shampoo_optimizer.py
Rscript techniques/shampoo-optimizer/r/shampoo_optimizer.R
```

**Refs:** Gupta, V., Koren, T. and Singer, Y. "Shampoo: preconditioned stochastic tensor optimization." In *ICML*, 2018; Anil, R. et al. "Scalable second order optimization for deep learning." arXiv:2002.09018, 2020.

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
