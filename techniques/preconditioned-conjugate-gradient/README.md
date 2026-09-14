# Preconditioned Conjugate Gradient (Reference §47.369)

Hestenes-Stiefel (1952) CG; Concus-Golub-O'Leary (1976) PCG
formulation. For SPD `A` and preconditioner `M` approximating
`A⁻¹`, run CG on the preconditioned system `M A x = M b`:

```
r ← b − A x
z ← M r
p ← z
loop:
    α = (r·z) / (p·A p)
    x ← x + α p
    r_new ← r − α A p
    z_new ← M r_new
    β = (r_new · z_new) / (r · z)
    p ← z_new + β p
```

Convergence rate depends on `κ(M A)`, so a good preconditioner
turns `O(√κ(A))` into `O(√κ(M A))`. Backbone of every serious
sparse SPD solver.

## Files

- `python/preconditioned_conjugate_gradient.py` — n=300 SPD
  matrix, eigenvalues from 1 to 1000 (κ = 1000). Plain CG
  and Jacobi PCG both take 124 iterations to reach `1e−10`
  residual; a "perfect" preconditioner (Cholesky of A)
  converges in **1 iteration** to `8e−14` (analytically CG
  finishes in one step when the preconditioned operator is
  the identity).
- `r/preconditioned_conjugate_gradient.R` —
  `Rlinsolve::lsolve.pcg`, `Matrix::solve` (R);
  `scipy.sparse.linalg.cg` with `LinearOperator`
  preconditioner, `pyamg`, from-scratch (Python).

## When to use

- **Large sparse SPD systems** — finite-element stiffness,
  graph Laplacian, kernel-ridge normal equations, GP
  Cholesky.
- **When a good preconditioner is available** — ILU, block
  Jacobi, algebraic multigrid, wavelet-based, deflation.
- **Matrix-free operators** — the CG kernel only needs
  `A @ v` and `M_solve(r)`.

## When NOT to use

- **Non-symmetric or indefinite systems** — use GMRES / BiCGStab /
  MINRES.
- **Small dense matrices** — direct Cholesky is trivial.
- **When no preconditioner is available and κ is huge** —
  convergence stalls.

## Assumptions & caveats

- **Preconditioner must be SPD** — else PCG breaks down; use
  MINRES if only symmetric.
- **Cost per iteration** — one `A @ v` + one `M_solve` +
  a few dots; often the `M_solve` dominates.
- **Stopping criterion** — relative or absolute residual;
  the true error can be a few orders larger than the
  residual for very ill-conditioned systems.
- **Loss of orthogonality** — CG in exact arithmetic
  finishes in n steps; in floating point, more due to
  round-off.
- **Deflation / recycling Krylov** — for sequences of
  related linear systems.

## Related in this repo

- `conjugate-gradient-cg` — un-preconditioned SPD baseline.
- `gmres-krylov-solver` — non-symmetric cousin.
- `lanczos-iteration` — related eigenvalue algorithm.
- `gaussian-process-regression` — a big application area
  (nearly every GP library uses PCG or a Cholesky variant).

## Run

```
python techniques/preconditioned-conjugate-gradient/python/preconditioned_conjugate_gradient.py
Rscript techniques/preconditioned-conjugate-gradient/r/preconditioned_conjugate_gradient.R
```

**Refs:** Hestenes, M.R. and Stiefel, E. "Methods of conjugate gradients for solving linear systems." *J. Res. Natl. Bur. Stand.*, 49(6): 409-436, 1952; Concus, P., Golub, G.H. and O'Leary, D.P. "A generalized conjugate gradient method for the numerical solution of elliptic partial differential equations." In *Sparse Matrix Computations*, Academic Press, 1976.

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
