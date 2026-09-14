# Conjugate Gradient (Reference §47.320)

Hestenes & Stiefel (1952). Iterative method for solving
`A x = b` with `A` symmetric positive-definite. Iterates
traverse CONJUGATE (A-orthogonal) directions:

```
r_0 = b − A x_0
p_0 = r_0
for k = 0, 1, …:
    α_k = (r_kᵀ r_k) / (p_kᵀ A p_k)
    x_{k+1} = x_k + α_k p_k
    r_{k+1} = r_k − α_k A p_k
    β_k = (r_{k+1}ᵀ r_{k+1}) / (r_kᵀ r_k)
    p_{k+1} = r_{k+1} + β_k p_k
```

Converges in ≤ n iterations exactly; much faster if A has
clustered eigenvalues.

## Files

- `python/conjugate_gradient_cg.py` — CG on a random n=100
  SPD system (M Mᵀ + n I). Converges in 37 iterations to
  residual 1e-10; ‖x_CG − x_direct‖ ~1e-9. Also: A with 2
  eigenvalue clusters converges in 2 iterations, illustrating
  the classical cluster-count bound.
- `r/conjugate_gradient_cg.R` — `Matrix::CHMfactor`, RcppEigen
  (R); `scipy.sparse.linalg.cg`, `torch.linalg.solve`,
  from-scratch (Python).

## When to use

- **Large sparse SPD systems** — CG is the workhorse.
- **Krylov-subspace methods** — the theoretical foundation.
- **Newton-CG optimisation** — inner solver for Hessian
  systems.
- **Preconditioned CG (PCG)** — accelerates convergence
  dramatically.

## When NOT to use

- **Non-symmetric / indefinite `A`** — use GMRES, BiCG,
  BiCGStab, MINRES.
- **Small dense systems** — direct solve is simpler and
  robust.
- **Ill-conditioned without preconditioner** — iterations
  can stall.

## Assumptions & caveats

- **Symmetric positive-definite `A`** — required.
- **Preconditioning** — Jacobi / IC / AMG dramatically speed
  convergence.
- **Floating-point issues** — CG can lose orthogonality; use
  reorthogonalisation for high accuracy.
- **Sparse-matrix-vector product cost** — CG is only useful
  if `A x` is cheap.

## Related in this repo

- `lbfgs-quasi-newton` — nonlinear generalisation.
- `newton-raphson` (via `wald-lrt-score` neighbourhood) —
  full-Hessian variant.
- `frank-wolfe-conditional-gradient` — LP-oracle alternative.

## Run

```
python techniques/conjugate-gradient-cg/python/conjugate_gradient_cg.py
Rscript techniques/conjugate-gradient-cg/r/conjugate_gradient_cg.R
```

**Refs:** Hestenes, M.R. and Stiefel, E. "Methods of conjugate gradients for solving linear systems." *J. Res. Nat. Bur. Standards*, 49(6): 409-436, 1952; Shewchuk, J.R. *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain*, CMU technical note, 1994.

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
