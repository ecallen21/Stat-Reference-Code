# Frank-Wolfe (Conditional Gradient) (Reference §47.100)

Frank & Wolfe (1956). For a differentiable convex f over a compact
convex set D:

    1. Linearise: s_k = argmin_{s ∈ D} ⟨∇f(x_k), s⟩.
    2. Step:      x_{k+1} = (1 − γ_k) x_k + γ_k s_k,  γ_k ∈ [0, 1].

**Projection-free** — s_k comes from a Linear Minimisation Oracle
(LMO). Convergence O(1/k). Popular for constrained problems where
LMO is cheap (L1 ball → single non-zero, nuclear-norm ball →
top singular vector).

## Files

- `python/frank_wolfe_conditional_gradient.py` — Frank-Wolfe for
  `min 0.5‖y − Xβ‖² s.t. ‖β‖₁ ≤ r`, exact line search on the
  quadratic. Demo (n=200, p=30, true support {0, 5, 12}):
  - r=1.0  → converges 16 iters, ||β||₁ = 1.0 (constraint-active)
  - r=10.0 → 500 iters, ||β||₁ ≈ 4.1, β̂ ≈ (1.56, −1.01, 0.79)
    at true positions.
  Iterate sparsity ↑ with each iteration (one active coord per step).
- `r/frank_wolfe_conditional_gradient.R` — `CVXR`,
  `cvxopt` (R); `cvxpy`, `pymanopt`, custom (Python).

## When to use

- **Convex constrained optimisation** with a cheap LMO.
- **Sparse / low-rank problems** — L1 or nuclear-norm ball →
  sparse iterates by construction.
- **Structured atomic decomposition** — Jaggi 2013 unification.
- **Streaming / large-scale** — LMO often cheaper than projection.

## When NOT to use

- **When projections are cheap** (box constraints, simplex) —
  projected-gradient / accelerated methods faster.
- **Non-convex problems** — no convergence guarantees.
- **Stiff Hessian problems** — plain FW is O(1/k); use
  Away-Steps FW or Pairwise FW for faster linear rates on
  polytopes.

## Assumptions & caveats

- **Convex f** and **compact convex D** for guarantees.
- **Zig-zagging** — vanilla FW slow on polytope corners; Away-Steps
  variant (Jaggi 2013) mitigates.
- **Step size** — `2/(k+2)` decaying is safe; line-search faster
  for smooth f.
- **LMO exactness** — inexact LMOs (block-FW, randomised) still work
  with weaker rates.

## Related in this repo

- `admm-consensus`, `coordinate-descent-lasso`,
  `ridge-lasso-elasticnet`, `group-lasso`,
  `dantzig-selector`, `stability-selection` — sparse-optim family.
- `matrix-completion-svt`, `robust-pca`, `sparse-pca` —
  nuclear-norm cousins (FW's top-singular-vector LMO shines).
- `distributionally-robust-optimization`,
  `bayesian-optimization` — constrained-opt neighbours.
- `optimal-transport-wasserstein` — LP LMO example.

## Run

```
python techniques/frank-wolfe-conditional-gradient/python/frank_wolfe_conditional_gradient.py
Rscript techniques/frank-wolfe-conditional-gradient/r/frank_wolfe_conditional_gradient.R
```

**Refs:** Frank, M. & Wolfe, P. "An algorithm for quadratic programming." *Naval Research Logistics* 3(1-2): 95-110, 1956; Jaggi, M. "Revisiting Frank-Wolfe: Projection-free sparse convex optimization." *ICML*, 2013.

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
