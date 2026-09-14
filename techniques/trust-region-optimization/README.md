# Trust-Region Optimisation (Reference §47.365)

Powell (1970); Steihaug (1983); Byrd-Schnabel (2000); Conn-
Gould-Toint (2000). At each iterate solve the quadratic model
sub-problem inside a trust-region ball of radius `Δ_k`:

```
min_p  m_k(p) = f_k + g_kᵀ p + ½ pᵀ H_k p
s.t.   ‖p‖ ≤ Δ_k
```

Update `Δ_k` based on the ratio `ρ_k = (f(x) − f(x+p)) / (m_k(0) − m_k(p))`:
if `ρ < 0.25` shrink; if `> 0.75` on the boundary, grow. Accept
step when `ρ > η`. Convergence is guaranteed even from bad
starting points, unlike line search.

## Files

- `python/trust_region_optimization.py` — Powell 1970
  DOG-LEG sub-solver (Cauchy point + Newton step
  interpolation). Rosenbrock at `x₀ = −1.2` in d=2, 5, 10:
  reaches machine precision (`f < 1e−18`, `‖x* − 1‖ < 5e−9`)
  in 24–36 iterations without any tuning.
- `r/trust_region_optimization.R` — `trust::trust`,
  `nloptr` (R); `scipy.optimize.minimize` methods `trust-ncg`,
  `trust-krylov`, `dogleg`, from-scratch (Python).

## When to use

- **Non-convex smooth objectives** — trust region gives global
  convergence guarantees line search does not.
- **When Hessian is available** — Newton-type direction inside
  the trust region.
- **Robust optimisation defaults** — `scipy.optimize`'s
  `least_squares` uses trust-region reflective for bound-
  constrained non-linear LSQ.

## When NOT to use

- **Very high-dim (n ≫ 10⁴)** — full Hessian is expensive;
  use L-BFGS or Krylov trust region (CG-Steihaug).
- **When each function evaluation is very cheap** — line
  search + BFGS is often faster in wall-clock.
- **Convex smooth problems** — Newton or L-BFGS + line
  search are simpler and equivalent.

## Assumptions & caveats

- **Sub-problem accuracy** — dog-leg is a cheap approximation;
  CG-Steihaug / GLTR are more accurate.
- **Δ update thresholds (η, 0.25, 0.75)** — Conn-Gould-Toint
  give convergence for a wide range; adaptive schemes exist.
- **Indefinite Hessian** — dog-leg needs positive definite
  Newton step; use CG-Steihaug or Moré-Sorensen for
  indefinite `H`.
- **Sub-Hessian scaling** — poor scaling degrades
  performance; use elliptical trust regions (Byrd-Schnabel).

## Related in this repo

- `conjugate-gradient-cg`, `lbfgs-quasi-newton`,
  `gauss-newton-lm-nlls` — related deterministic optimisation.
- `proximal-newton`, `nonlinear-least-squares` — trust-region
  is Standard for these.
- `mm-majorization-minimization`, `fista-accelerated-proximal`,
  `proximal-gradient-method` — first-order alternatives.
- `bayesian-optimization`, `simulated-annealing`,
  `nelder-mead-simplex` — derivative-free cousins.

## Run

```
python techniques/trust-region-optimization/python/trust_region_optimization.py
Rscript techniques/trust-region-optimization/r/trust_region_optimization.R
```

**Refs:** Powell, M.J.D. "A new algorithm for unconstrained optimization." In *Nonlinear Programming*, Academic Press, 1970; Conn, A.R., Gould, N.I.M. and Toint, P.L. *Trust-Region Methods*, SIAM, 2000; Byrd, R.H. and Schnabel, R.B. "A trust region method based on interior point techniques for nonlinear programming." *Math. Program.*, 89: 149-185, 2000.

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
