# GMRES — Generalized Minimum Residual (Reference §47.368)

Saad & Schultz (1986). Iterative solver for `A x = b` when
`A` is **non-symmetric**. Build an orthonormal Krylov basis
via ARNOLDI, then choose the iterate that MINIMISES the
2-norm of the residual over `x₀ + K_k`:

```
K_k(A, r₀) = span{r₀, A r₀, A² r₀, …, A^(k−1) r₀}
```

Givens-rotated upper-Hessenberg factorisation lets you monitor
the residual without solving the least-squares problem at each
step. **Restart every m steps** to bound memory (GMRES(m)).

## Files

- `python/gmres_krylov_solver.py` — Random non-symmetric
  diagonally-dominant `A ∈ ℝ^(200×200)`, 10% asymmetry.
  Clustered spectrum → 12 matvecs suffice for `1e−12`
  residual across restart sizes 10 / 30 / 60. Matches
  `numpy.linalg.solve` to relative error 2e−12.
- `r/gmres_krylov_solver.R` — `Rlinsolve::lsolve.gmres`,
  `pracma::gmres` (R); `scipy.sparse.linalg.gmres`,
  `scipy.sparse.linalg.lgmres`, from-scratch (Python).

## When to use

- **Large sparse non-symmetric linear systems** — PDEs
  (advection-diffusion), finite-element discretisations,
  power-flow equations.
- **Matrix-free operators** — only requires an `A @ v`
  routine, no explicit A.
- **Combined with a good preconditioner** — ILU / block
  Jacobi / algebraic multigrid dramatically cut iterations.

## When NOT to use

- **Symmetric positive definite systems** — CG is cheaper
  (no Arnoldi, three-term recurrence).
- **Nearly-singular / ill-conditioned without a
  preconditioner** — convergence stalls; residual can even
  stagnate.
- **Dense small systems** — direct factorisation is trivial.

## Assumptions & caveats

- **Convergence** — polynomial in the spectrum of the
  preconditioned operator; clustered eigenvalues far from 0
  ⇒ fast convergence.
- **Restart parameter m** — small m saves memory but may
  stagnate; adaptive `m` schemes exist.
- **Right vs left preconditioning** — different residual
  histories; right-preconditioning most common.
- **Loss of orthogonality** — modified Gram-Schmidt is
  standard; classical GS can lose 12+ digits.
- **Breakdown at exact solution** — Arnoldi has to detect
  `H[j+1, j] ≈ 0`.

## Related in this repo

- `conjugate-gradient-cg` — symmetric positive definite
  cousin.
- `preconditioned-conjugate-gradient` — SPD with
  preconditioner.
- `lanczos-iteration` — symmetric eigenvalue cousin.
- `bcg-biconjugate-gradient` (if present) — non-symmetric
  alternative with three-term recurrence.

## Run

```
python techniques/gmres-krylov-solver/python/gmres_krylov_solver.py
Rscript techniques/gmres-krylov-solver/r/gmres_krylov_solver.R
```

**Refs:** Saad, Y. and Schultz, M.H. "GMRES: a generalized minimal residual algorithm for solving nonsymmetric linear systems." *SIAM J. Sci. Stat. Comput.*, 7(3): 856-869, 1986; Saad, Y. *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.

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
