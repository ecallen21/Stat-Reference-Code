# Sequential Quadratic Programming (Reference §47.367)

Wilson (1963); Han (1976); Powell (1978). For a nonlinearly-
constrained problem `min f(x) s.t. h(x) = 0` (extends to
inequalities), solve a sequence of QP sub-problems around the
current iterate:

```
min_p  ∇f(x_k)ᵀ p + ½ pᵀ B_k p
s.t.   h(x_k) + J_h(x_k) p = 0
```

where `B_k` is a (damped-BFGS) approximation of the Hessian of
the LAGRANGIAN. Combined with a merit function
`φ(x) = f(x) + μ ‖h(x)‖₁` for globalisation.

## Files

- `python/sequential_quadratic_programming.py` — Rosenbrock
  constrained to the unit circle `x² + y² = 1`. Analytical
  minimum near `(0.7864, 0.6177), f* ≈ 0.0457`. From-scratch
  SQP with damped-BFGS and L1 merit hits `f = 0.045675,
  ‖h‖ = 1.1e−16` in 19 iterations, matching
  `scipy.optimize.minimize(method='SLSQP')` to 5 decimals.
- `r/sequential_quadratic_programming.R` — `nloptr::slsqp`,
  `Rsolnp`, `alabama::auglag` (R); `scipy.optimize` SLSQP,
  `cvxpy`, `casadi`, from-scratch (Python).

## When to use

- **Nonlinear constrained smooth optimisation** — MLE with
  equality constraints, calibration under conservation laws.
- **Small-to-medium problems** (n ≤ 10⁴) — the QP sub-problem
  is exactly solvable.
- **Model-fit / calibration with sum-to-one, moment
  constraints** — SQP is the go-to.

## When NOT to use

- **Very large problems** — QP subproblem cost dominates;
  use L-BFGS-B (bounds only) or interior-point.
- **Non-smooth constraints** — use ADMM / augmented
  Lagrangian.
- **Purely inequality constrained convex** — IPM is simpler
  and comes with polynomial complexity guarantees.

## Assumptions & caveats

- **Merit-function penalty μ** — must exceed the L∞-norm of
  the Lagrange multipliers; too small and the merit does not
  align with the KKT conditions.
- **Damped BFGS** — Powell's damping keeps `B_k ≻ 0`; skip
  it and B can become indefinite.
- **QP sub-problem infeasibility** — the linearised
  constraints can be infeasible; augmented Lagrangian
  variants handle this.
- **Second-order sufficient conditions** — SQP converges
  Q-superlinearly under them; slow otherwise.
- **Active-set for inequalities** — Powell's SQP handles
  inequality constraints via active-set QP; modern solvers
  (SNOPT, IPOPT) mix SQP + IPM.

## Related in this repo

- `interior-point-method`, `admm-consensus`,
  `frank-wolfe-conditional-gradient`, `proximal-newton` —
  constrained-optimisation neighbours.
- `trust-region-optimization`, `lbfgs-quasi-newton`,
  `gauss-newton-lm-nlls` — unconstrained cousins.
- `nonlinear-least-squares`, `nonlinear-mixed-effects` —
  common statistical applications.
- `optimal-transport-wasserstein`, `wasserstein-barycenter`
  — OT solvers often use SQP variants.

## Run

```
python techniques/sequential-quadratic-programming/python/sequential_quadratic_programming.py
Rscript techniques/sequential-quadratic-programming/r/sequential_quadratic_programming.R
```

**Refs:** Wilson, R.B. "A simplicial algorithm for concave programming." PhD thesis, Harvard, 1963; Han, S.-P. "Superlinearly convergent variable metric algorithms for general nonlinear programming problems." *Math. Program.*, 11: 263-282, 1976; Powell, M.J.D. "A fast algorithm for nonlinearly constrained optimization calculations." In *Numerical Analysis*, Springer LNM 630, 1978.

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
