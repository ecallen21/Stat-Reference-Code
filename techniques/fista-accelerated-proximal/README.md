# FISTA — Fast Iterative Shrinkage-Thresholding (Reference §47.318)

Beck & Teboulle (2009). Nesterov-accelerated version of ISTA /
proximal gradient. Adds a MOMENTUM step:

```
y_k = x_k + ((t_{k−1} − 1) / t_k)(x_k − x_{k−1})
x_{k+1} = prox_{η g}(y_k − η ∇f(y_k))
t_{k+1} = (1 + √(1 + 4 t_k²)) / 2
```

Achieves O(1/k²) convergence vs plain ISTA's O(1/k) at the
same per-iteration cost. Standard for LASSO / TV-regularised
image denoising.

## Files

- `python/fista_accelerated_proximal.py` — ISTA and FISTA
  side-by-side on a LASSO (n=200, d=100). FISTA gap to
  optimum decays orders faster: at iter 30 FISTA gap ≈ 6e-9
  vs ISTA gap ≈ 2e-8; at iter 100 FISTA is at machine
  precision.
- `r/fista_accelerated_proximal.R` — `celer` via reticulate,
  `glmnet` (R); `proxop`, `pyproximal`, `sklearn.Lasso`,
  from-scratch (Python).

## When to use

- **LASSO / total-variation / group-lasso** at scale.
- **Any composite convex problem** where ISTA converges too
  slowly.
- **Parallel-per-sample settings** — FISTA parallelises the
  gradient computation.

## When NOT to use

- **Strongly non-convex penalties** — FISTA's guarantee is
  convex; use MM for non-convex.
- **Highly ill-conditioned Hessians** — plain FISTA still
  suffers; use Nesterov's optimal method for strong convexity.
- **Restarting needed** — vanilla FISTA can oscillate; use
  restart schemes (O'Donoghue & Candès 2015).

## Assumptions & caveats

- **Step size** — same 1/L rule as ISTA.
- **Convergence proof** requires convex smooth `f` +
  convex proximable `g`.
- **Restart heuristics** improve empirical convergence when
  the objective is locally strongly convex.
- **Line search** (FISTA-BT) accepts larger steps when
  possible.

## Related in this repo

- `proximal-gradient-method` — the base (ISTA) algorithm.
- `nesterov-accelerated-gradient` — the smooth-case cousin.
- `adam-optimizer`, `rmsprop-optimizer` — adaptive-step
  alternatives.

## Run

```
python techniques/fista-accelerated-proximal/python/fista_accelerated_proximal.py
Rscript techniques/fista-accelerated-proximal/r/fista_accelerated_proximal.R
```

**Refs:** Beck, A. and Teboulle, M. "A fast iterative shrinkage-thresholding algorithm for linear inverse problems." *SIAM J. Imaging Sci.*, 2(1): 183-202, 2009.

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
