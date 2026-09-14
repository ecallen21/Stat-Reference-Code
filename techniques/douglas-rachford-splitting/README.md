# Douglas-Rachford Splitting (Reference §47.380)

Douglas & Rachford (1956); modern proximal form
Combettes-Wajs (2005). Solve `min_x f(x) + g(x)` using only
the proximal operators of `f` and `g`:

```
y_k = prox_{λ f}(z_k)
z_{k+1} = z_k + prox_{λ g}(2 y_k − z_k) − y_k
```

Fixed points lie in the minimiser set of `f + g`. Reduces
composite optimisation to two prox evaluations per iteration
— powerful when neither `f` nor `g` is smooth. **ADMM is a
special case** applied to the dual of a linearly-constrained
problem.

## Files

- `python/douglas_rachford_splitting.py` — LASSO
  `min 0.5 ‖Ax − b‖² + λ ‖x‖₁` (n=200, d=100, sparsity 10).
  DR converges to the same objective (9.31) as
  `sklearn.linear_model.Lasso`; support sizes match (61 vs
  59, both a mix of true signal + noise-active).
- `r/douglas_rachford_splitting.R` — `CVXR` (uses DR
  splitting internally), `flare`, custom loop (R);
  `pyproximal.DouglasRachford`, `cvxpy`, from-scratch
  (Python).

## When to use

- **Composite non-smooth optimisation** — LASSO, group
  LASSO, TV denoising, total-variation image reconstruction.
- **When only PROX is available** — closed-form prox exists
  for L1, L2², nuclear-norm, indicator functions of common
  cones.
- **Distributed / consensus optimisation** — ADMM is DR on
  the dual and parallelises across nodes.

## When NOT to use

- **Both terms smooth** — plain gradient / L-BFGS is simpler
  and faster.
- **Very large linear operators** — prox of the quadratic
  requires a system solve; use LSQR-based inner solves or
  primal-dual algorithms.
- **When adaptive step size is unavailable** — Chambolle-
  Pock and BB-DR variants help.

## Assumptions & caveats

- **λ parameter** — stability requires `λ > 0`; too small
  ⇒ slow convergence, too large ⇒ divergence in some
  variants (though DR is unconditionally stable on convex).
- **Prox availability** — closed-form prox for L1, L2², group
  LASSO, box constraints; inner solves needed otherwise.
- **Convergence rate** — sublinear `O(1/k)` in general;
  linear under strong convexity of one summand.
- **Fixed-point vs primal** — the iterate `z_k` is not the
  primal solution; recover `x* = prox_{λ f}(z*)`.
- **ADMM equivalence** — DR on the dual = ADMM on the primal;
  penalty ρ ↔ 1/λ.

## Related in this repo

- `proximal-gradient-method`, `fista-accelerated-proximal`,
  `admm-consensus`, `frank-wolfe-conditional-gradient`,
  `proximal-newton` — splitting neighbours.
- `chambolle-pock-primal-dual` — primal-dual cousin for
  saddle-point problems.
- `ridge-lasso-elasticnet`, `group-lasso`, `fused-lasso` —
  statistical applications.
- `total-variation-denoising` (if present), `robust-pca` —
  image-processing applications.

## Run

```
python techniques/douglas-rachford-splitting/python/douglas_rachford_splitting.py
Rscript techniques/douglas-rachford-splitting/r/douglas_rachford_splitting.R
```

**Refs:** Douglas, J. and Rachford, H.H. "On the numerical solution of heat conduction problems in two and three space variables." *Trans. Amer. Math. Soc.*, 82: 421-439, 1956; Combettes, P.L. and Wajs, V.R. "Signal recovery by proximal forward-backward splitting." *SIAM Multiscale Model. Simul.*, 4(4): 1168-1200, 2005.

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
