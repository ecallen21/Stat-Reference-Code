# Interior-Point Method (Reference §47.366)

Karmarkar (1984); Nesterov-Nemirovski (1994). For an
inequality-constrained convex program

```
min f(x)   subject to   A x ≤ b
```

replace hard constraints by a LOG BARRIER

```
F(x, t) = t · f(x) − Σ_i log(b_i − a_iᵀ x)
```

and minimise `F` for a sequence of increasing `t → ∞`.
Iterates stay STRICTLY FEASIBLE. For self-concordant barriers
the convergence bound is `O(√m · log(1/ε))` Newton steps.

## Files

- `python/interior_point_method.py` — classical 2-D LP
  `min −3x₁ − 5x₂ s.t. x₁ ≤ 4, 2x₂ ≤ 12, 3x₁ + 2x₂ ≤ 18,
  x ≥ 0`. Interior-point recovers `x* = (2, 6), f* = −36` to
  machine precision and matches `scipy.linprog(method='highs')`.
- `r/interior_point_method.R` — `CVXR`, `Rglpk`, `lpSolve`
  (R); `scipy.optimize.linprog(method='highs-ipm')`,
  `cvxpy` (ECOS/SCS/Clarabel are IPMs), from-scratch (Python).

## When to use

- **Linear programs (LP)** — HIGHS / IPOPT / MOSEK all use
  IPM; simplex is the alternative.
- **Convex quadratic programs (QP)** — the standard solver
  in `quadprog`, `OSQP`, `CVXPY`.
- **Second-order cone / semi-definite programs (SOCP / SDP)**
  — self-concordant barriers extend naturally.
- **Support vector machines (large)** — sequential-minimal-
  optimisation-vs-IPM trade-off; IPM wins on some problems.

## When NOT to use

- **Non-convex problems** — barrier is only defined inside
  the feasible set; use trust region / SQP.
- **Very large sparse LPs where simplex is faster** — network
  flow, transportation, some real-world scheduling.
- **When solutions must be integer** — LP relaxation + IPM +
  rounding often bad; use branch-and-bound MILP.

## Assumptions & caveats

- **Strictly feasible start** — need `A x⁰ < b`; use
  Phase-I LP if unknown.
- **Central-path parameter** — `μ = 10` is standard for
  `t ← μ · t`; adaptive schedules are common.
- **Backtracking line search** — needed to stay inside the
  feasible set.
- **Predictor-corrector IPM** — Mehrotra's algorithm is the
  practical variant for LP / QP.
- **Numerical conditioning** — near the boundary the Hessian
  becomes ill-conditioned; iterative refinement helps.

## Related in this repo

- `proximal-newton`, `sequential-quadratic-programming`,
  `frank-wolfe-conditional-gradient`, `admm-consensus` —
  constrained-optimisation neighbours.
- `trust-region-optimization`, `lbfgs-quasi-newton`,
  `conjugate-gradient-cg` — unconstrained cousins.
- `svm-classifier`, `support-vector-regression` — IPM is
  one training path.
- `linear-programming-simplex` (if present), `quadprog`
  wrappers — LP/QP solvers.

## Run

```
python techniques/interior-point-method/python/interior_point_method.py
Rscript techniques/interior-point-method/r/interior_point_method.R
```

**Refs:** Karmarkar, N. "A new polynomial-time algorithm for linear programming." *Combinatorica*, 4: 373-395, 1984; Nesterov, Y. and Nemirovski, A. *Interior-Point Polynomial Algorithms in Convex Programming*, SIAM, 1994; Boyd, S. and Vandenberghe, L. *Convex Optimization*, Cambridge Univ Press, 2004.

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
