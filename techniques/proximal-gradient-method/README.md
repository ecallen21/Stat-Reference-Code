# Proximal Gradient Method (Reference §47.317)

Combettes & Wajs (2005); Parikh & Boyd (2014). Minimise the
composite objective `f(x) + g(x)` where `f` is smooth and `g`
is a possibly non-smooth "simple" regulariser:

```
x_{k+1} = prox_{η g}(x_k − η ∇f(x_k))
prox_{η g}(v) = argmin_z (½‖z − v‖² + η g(z))
```

For `g(x) = λ‖x‖₁` the prox is SOFT-THRESHOLDING; the whole
scheme is then ISTA. Extends to a huge family of regularisers
(L1, group-lasso, total variation, box constraints) via
closed-form proxes.

## Files

- `python/proximal_gradient_method.py` — ISTA on a LASSO with
  n=200, d=50, 5 true nonzeros, λ=0.3. Recovers exactly 5
  nonzeros (top-5 magnitudes match truth ≈ 3, 2, 2, 1.5, 0.5).
  Objective plateaus by ~50 iterations at ~2.58.
- `r/proximal_gradient_method.R` — `glmnet` (coord descent),
  `ncvreg`, `admm.lasso` (R); `sklearn.linear_model.Lasso`,
  `celer`, `proxop` / `pyproximal`, from-scratch (Python).

## When to use

- **Non-smooth regularised regression** — LASSO, elastic net,
  group lasso, TV.
- **Constrained problems** — box constraints have simple
  projection prox.
- **Large-scale** — cheap per-iteration cost, memory-friendly.

## When NOT to use

- **Very ill-conditioned smooth part** — plain proximal
  gradient is slow; use FISTA / L-BFGS-B.
- **When solving a fully differentiable objective** — L-BFGS
  is faster.
- **Non-convex penalties (SCAD, MCP)** — need a majorization
  step or a specialised solver.

## Assumptions & caveats

- **Learning rate η ≤ 1 / L** (L = Lipschitz constant of ∇f);
  the demo uses 1 / λ_max(XᵀX / n).
- **Prox operator** — must be efficient; ISTA depends on
  closed-form soft-threshold.
- **Convergence rate** — plain ISTA is O(1/k); FISTA is O(1/k²).
- **Coordinate descent** (`glmnet`) is usually faster than
  ISTA in practice for LASSO with small feature counts.

## Related in this repo

- `fista-accelerated-proximal` — the accelerated companion.
- `mm-majorization-minimization` — the umbrella framework.
- `admm-consensus` — dual-decomposition alternative.
- `adaptive-lasso`, `fused-lasso`, `group-lasso`, `scad-mcp-penalties`
  — regularisers this framework handles.

## Run

```
python techniques/proximal-gradient-method/python/proximal_gradient_method.py
Rscript techniques/proximal-gradient-method/r/proximal_gradient_method.R
```

**Refs:** Combettes, P.L. and Wajs, V.R. "Signal recovery by proximal forward-backward splitting." *SIAM Multiscale Model. Simul.*, 4(4): 1168-1200, 2005; Parikh, N. and Boyd, S. "Proximal Algorithms." *Found. Trends Optim.*, 1(3): 127-239, 2014.

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
