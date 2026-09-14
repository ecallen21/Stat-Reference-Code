# SAG / SAGA — Variance-Reduced SGD (Reference §47.337)

Roux, Schmidt & Bach (2012, SAG); Defazio, Bach & Lacoste-
Julien (2014, SAGA). Store per-sample gradients and use their
AVERAGE as the search direction:

```
draw i;   d = ∇_i f(x) − g_i + mean(g)
x <- x − η d
g_i <- ∇_i f(x)     (update memory)
```

Achieves LINEAR convergence on strongly convex objectives at
SGD-per-step cost, unlike vanilla SGD (sub-linear).

## Files

- `python/sag_saga_variance_reduction.py` — SAGA on ridge
  regression with per-sample memory. n=200, d=20. SAGA drives
  the gap to the closed-form ridge solution to ~ 3e-5 by
  10 epochs; SGD (constant lr) stalls at ~ 1e-2.
- `r/sag_saga_variance_reduction.R` — reticulate + sklearn,
  `glmnet` (R); `sklearn.linear_model.LogisticRegression(solver='saga')`,
  `lightning` classifiers, from-scratch (Python).

## When to use

- **Strongly convex + smooth** objectives (ridge, logistic
  ridge) with n moderate.
- **Sample-averaged gradient available** — SGD in the same
  cost regime but with linear convergence.
- **Memory affordable** — SAGA stores O(n·d) gradients.

## When NOT to use

- **Very large n or d** — memory prohibitive; use SVRG
  (variance-reduction without memory) or vanilla SGD.
- **Non-convex** — variance reduction helps less; use Adam /
  RMSProp.
- **Deep learning** — variance-reduced methods have not
  dominated SGD+momentum in practice for large-scale DL.

## Assumptions & caveats

- **Strong convexity** — the exponential-convergence rate
  requires it; without SC, get O(1/k) which still beats SGD.
- **Step size η ≤ 1/(3L)** — SAGA is more forgiving than SAG.
- **Initial burn-in** — store one full gradient pass at the
  start (SAGA variant).
- **Sparse gradients** — SAGA has lazy update variants for
  sparse features.

## Related in this repo

- `online-learning-sgd`, `adam-optimizer`,
  `rmsprop-optimizer`, `nesterov-accelerated-gradient` —
  first-order alternatives.
- `stochastic-gradient-mcmc` — related randomised optimisation.
- `lbfgs-quasi-newton` — deterministic second-order.

## Run

```
python techniques/sag-saga-variance-reduction/python/sag_saga_variance_reduction.py
Rscript techniques/sag-saga-variance-reduction/r/sag_saga_variance_reduction.R
```

**Refs:** Roux, N., Schmidt, M. and Bach, F. "A stochastic gradient method with an exponential convergence rate for strongly convex optimization." In *NeurIPS*, 2012; Defazio, A., Bach, F. and Lacoste-Julien, S. "SAGA: A fast incremental gradient method with support for non-strongly convex composite objectives." In *NeurIPS*, 2014.

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
