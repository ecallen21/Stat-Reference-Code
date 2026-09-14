# LinUCB Contextual Bandit (Reference §47.305)

Li, Chu, Langford & Schapire (2010). Per-arm ridge regression
with an upper-confidence-bound acquisition:

```
A_a = I + Σ x xᵀ
θ̂_a = A_a⁻¹ Σ x r
UCB_a(x) = θ̂_aᵀ x + α · √(xᵀ A_a⁻¹ x)
```

Pick argmax UCB_a. Extends UCB / Thompson from multi-armed
to CONTEXTUAL bandits with a per-arm linear reward model.

## Files

- `python/linucb_contextual_bandit.py` — 4-arm, 5-D context,
  T=2000. Cumulative regret grows sub-linearly (~27 by t=2000)
  vs the linear regret of random play; matches the LinUCB
  O(d √T log T) bound.
- `r/linucb_contextual_bandit.R` — `contextual` (R),
  reticulate + `MABWiser` / `contextualbandits` (R);
  `contextualbandits`, `vowpalwabbit`, `MABWiser`,
  from-scratch (Python).

## When to use

- **Personalisation** with contextual features (recommenders,
  news article selection).
- **A/B trees with rich covariates** — Thompson / UCB with
  side information.
- **Cold-start recommendation** — LinUCB starts safely with
  the identity prior.

## When NOT to use

- **Non-linear reward** — need Neural-LinUCB / GP-UCB.
- **Very high-dimensional context** — A_a inverse is O(d³).
- **Delayed / batched feedback** — vanilla LinUCB assumes
  immediate reward; extensions exist for delay.

## Assumptions & caveats

- **Reward = θ_aᵀ x + noise** — linear model per arm.
- **α (exploration bonus)** — 1.0 is default; theory pinpoints
  α ≥ 1 + √(ln(2/δ)/2).
- **Shared vs disjoint parameters** — LinUCB-hybrid (Li 2010)
  shares part of θ across arms.
- **Warm-start** — a few rounds of ε-greedy help before pure
  UCB.

## Related in this repo

- `multi-armed-bandits`, `thompson-sampling` — non-contextual
  baselines.
- `bayesian-optimization` — Gaussian-process analogue.
- `experimentation-platform`, `bayesian-ab-testing` — related
  A/B / online-learning designs.

## Run

```
python techniques/linucb-contextual-bandit/python/linucb_contextual_bandit.py
Rscript techniques/linucb-contextual-bandit/r/linucb_contextual_bandit.R
```

**Refs:** Li, L., Chu, W., Langford, J. and Schapire, R.E. "A contextual-bandit approach to personalized news article recommendation." In *WWW*, 2010.

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
