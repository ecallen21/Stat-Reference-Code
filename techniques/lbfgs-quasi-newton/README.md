# L-BFGS — Limited-Memory Quasi-Newton (Reference §47.319)

Liu & Nocedal (1989); Byrd et al (1995, L-BFGS-B). Quasi-Newton
method maintaining only the last `m` (s_k, y_k) pairs to
approximate the inverse Hessian:

```
s_k = x_{k+1} − x_k
y_k = ∇_{k+1} − ∇_k
H_{k+1} ≈ two-loop recursion over stored pairs
```

Standard for smooth unconstrained optimisation up to ~10⁷
parameters. L-BFGS-B adds box constraints.

## Files

- `python/lbfgs_quasi_newton.py` — Two-loop recursion + basic
  backtracking line search. Demo: fit ridge regression by
  L-BFGS. Converges in 15 iterations; distance from
  closed-form ridge solution ≈ 2.7e-7.
- `r/lbfgs_quasi_newton.R` — `stats::optim('L-BFGS-B')`,
  `lbfgs`, `optimParallel` (R); `scipy.optimize.minimize`,
  `torch.optim.LBFGS`, from-scratch (Python).

## When to use

- **Smooth unconstrained / box-constrained** optimisation.
- **Mid-to-large scale** (up to ~10⁷ parameters).
- **Fewer Hessian evaluations** than Newton — memory
  O(m · d) with m = 5-20.

## When NOT to use

- **Non-smooth objectives** — use proximal gradient / ADMM.
- **Very ill-conditioned Hessians** — L-BFGS's rank-m
  Hessian approximation misfits; use full Newton or CG.
- **Constrained optimisation beyond box** — use trust-region
  or interior-point methods.

## Assumptions & caveats

- **Memory m** — 5-20 typical; larger = better Hessian
  approximation but more RAM.
- **Line search** — Wolfe conditions recommended; strict
  Armijo can stall.
- **Curvature condition** — `s_k · y_k > 0` required to
  update H; skip pair when violated.
- **Batching for ML** — L-BFGS is deterministic; for
  stochastic optimisation see stochastic L-BFGS (Byrd 2016).

## Related in this repo

- `adam-optimizer`, `rmsprop-optimizer`, `nesterov-accelerated-gradient`
  — first-order alternatives.
- `conjugate-gradient-cg` — SPD linear systems / smooth
  quadratic minimisation.
- `gauss-newton-lm-nlls` — nonlinear least squares
  specialisation.

## Run

```
python techniques/lbfgs-quasi-newton/python/lbfgs_quasi_newton.py
Rscript techniques/lbfgs-quasi-newton/r/lbfgs_quasi_newton.R
```

**Refs:** Liu, D.C. and Nocedal, J. "On the limited memory BFGS method for large scale optimization." *Math. Program.*, 45: 503-528, 1989; Byrd, R.H., Lu, P., Nocedal, J. and Zhu, C. "A limited memory algorithm for bound constrained optimization." *SIAM J. Sci. Comput.*, 16(5): 1190-1208, 1995.

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
