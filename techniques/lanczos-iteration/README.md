# Lanczos Iteration (Reference §47.343)

Lanczos (1950). Build a Krylov subspace
`K_k(A, v) = span{v, Av, A²v, …, A^(k−1)v}` orthogonalised
by three-term recurrence:

```
β_j v_j = A v_{j−1} − α_{j−1} v_{j−1} − β_{j−1} v_{j−2}
α_j     = v_jᵀ A v_j
```

The projected matrix `T = V_kᵀ A V_k` is TRIDIAGONAL; its
eigenvalues (Ritz values) converge to extremal eigenvalues of
A in `O(k)` matrix-vector products. Backbone of ARPACK,
`scipy.sparse.linalg.eigsh`, PageRank power methods, and
implicit-Q shifts.

## Files

- `python/lanczos_iteration.py` — 400×400 symmetric matrix.
  `k = 10` gives coarse Ritz values; `k = 30` matches top-3
  eigenvalues to 0.12; `k = 60` matches to 5e-6. Cost is
  linear in `k` per matvec, vs the O(n³) exact eigh.
- `r/lanczos_iteration.R` — `RSpectra::eigs_sym` (ARPACK
  IRLM) (R); `scipy.sparse.linalg.eigsh`, `torch.lobpcg`,
  from-scratch (Python).

## When to use

- **Very large sparse or matvec-only symmetric matrices** —
  graph Laplacians, kernel matrices with fast product.
- **Top-k eigenvalues** or **smallest-k** by shift-invert.
- **Spectral graph theory, MCMC mixing analysis, PDE eigen-
  problems**.

## When NOT to use

- **Interior eigenvalues without shift-invert** — Lanczos
  favours extreme values; use shift-invert or LOBPCG.
- **Non-symmetric matrices** — use Arnoldi (Lanczos's non-
  symmetric cousin).
- **Full spectrum needed** — use `eigh` / `eig` (O(n³)).

## Assumptions & caveats

- **Re-orthogonalisation** — floating-point loss of
  orthogonality is severe; full or selective
  re-orthogonalisation is standard.
- **Ghost eigenvalues** — repeated Ritz values without
  re-orthogonalisation.
- **Convergence rate** — geometric in the gap ratio between
  the target and unwanted eigenvalues; Kaniel-Paige-Saad
  bounds.
- **Implicitly restarted Lanczos (IRLM)** — ARPACK's version;
  fixes memory blow-up for large k.

## Related in this repo

- `randomized-svd` — probabilistic alternative for singular
  triplets.
- `conjugate-gradient-cg` — Krylov cousin for linear systems.
- `nystrom-approximation`, `random-fourier-features` —
  low-rank kernel approximations.
- `pca`, `kernel-pca`, `spectral-clustering` — downstream
  spectral tasks.

## Run

```
python techniques/lanczos-iteration/python/lanczos_iteration.py
Rscript techniques/lanczos-iteration/r/lanczos_iteration.R
```

**Refs:** Lanczos, C. "An iteration method for the solution of the eigenvalue problem of linear differential and integral operators." *J. Res. Natl. Bur. Stand.*, 45: 255-282, 1950.

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
