# Distributionally Robust Optimisation — Group DRO (Reference Ch 30 Robustness)

**Minimise the max group loss, not the average.** Sagawa, Koh, Hashimoto
& Liang (2020) proposed Group DRO as the standard approach when
subgroups matter — spurious correlations that dominate the majority
population can catastrophically fail on a minority group under ERM.

## Objective

Standard ERM:

```
min_θ   𝔼_{(x,y) ~ P}  L(f_θ(x), y)
```

Group DRO (with known groups `g = 1..G`):

```
min_θ   max_g   𝔼_{(x,y) | g}  L(f_θ(x), y)
```

## Online algorithm (Sagawa 2020)

```
Initialise q ∝ 1/G.
Repeat:
    for each batch b:
        loss_g = mean loss on group g in batch b       (for all g)
        q_g   ← q_g · exp( η_q · loss_g )              (exponentiated ascent)
        q     ← q / Σ q
        Take an SGD step on Σ_g q_g · loss_g w.r.t. θ.
```

`η_q` is the group-weight step size; typical `η_q ∈ [0.01, 0.1]`.

## When to use

- **Known subgroups** (race / sex / hospital / site / language) where the
  worst-group performance must be bounded.
- **Spurious correlations** — the classic Waterbirds and CelebA-Blond
  benchmarks; the demo here mimics the pattern with a synthetic
  minority whose spurious feature disagrees with the majority.
- **Domain generalisation** — treat each training domain as a "group".

## When NOT to use

- **Group labels unavailable** — see JTT (Just Train Twice) and Colored-
  MNIST-style unsupervised methods.
- **All groups equally easy** — Group DRO gives no benefit; ERM is fine.
- **Very small minority groups** — the exponentiated weights explode
  variance; regularise with a max-`q` cap.

## Files

- `python/distributionally_robust_optimization.py` — from-scratch Group-
  DRO for logistic regression. Synthetic 3-feature problem where a
  spurious feature `x₁` is correlated with `y` in the majority group
  (n=800) and *anti-correlated* in a minority group (n=100). Result:
  **ERM 97 % majority acc but only 27 % minority acc** (relying on
  spurious feature); **DRO 75 % / 80 %** — a 53-point minority gain.
- `r/distributionally_robust_optimization.R` — `reticulate` + `wilds` /
  `fairlearn`; native R via `fairml` / `mlr3fairness`.

## Assumptions & caveats

- **Groups must be known at training time.** Weakly-supervised versions
  exist (JTT, Sohoni-EDA, GEORGE).
- **η_q sensitive** — too high oscillates; too low never re-weights.
- **Regularise the model** — Sagawa 2020 emphasises that Group DRO needs
  strong L2 (or dropout) to actually improve worst-group accuracy;
  otherwise it just memorises the minority.
- **Not a fairness cure-all** — Group DRO minimises worst-group risk,
  not e.g. demographic parity. See `fairness` techniques for other
  criteria.
- **CVaR DRO / χ²-DRO** relax the max to a chance constraint (Duchi-
  Namkoong 2019); useful when groups are latent.

## Related in this repo

- `covariate-shift-adaptation` — density-ratio weighting when the
  shift is at the *feature* level, not the group level.
- `class-imbalance` — a special case with groups defined by label.
- `fairness-metrics` / `fairness-mitigation` (future batches) — the
  broader family DRO belongs to.
- `logistic-regression`, `deep-mlp-backprop` — the base models.

## Run

```
python techniques/distributionally-robust-optimization/python/distributionally_robust_optimization.py
Rscript techniques/distributionally-robust-optimization/r/distributionally_robust_optimization.R
```

**Refs:** Sagawa, S., Koh, P.W., Hashimoto, T. & Liang, P. "Distributionally robust neural networks for group shifts." *ICLR*, 2020; Duchi, J. & Namkoong, H. "Learning models with uniform performance via distributionally robust optimisation." arXiv:1810.08750, 2019; Liu, E.Z. et al. "Just train twice: improving group robustness without training group information." *ICML*, 2021.

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
