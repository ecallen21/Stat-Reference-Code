# Proximal Newton (Reference §47.340)

Lee, Sun & Saunders (2014). Composite optimisation `f + g`
with SECOND-ORDER info of `f`:

```
x_{k+1} = argmin_z ⟨∇f(x_k), z − x_k⟩
             + ½ (z − x_k)ᵀ H_k (z − x_k)
             + g(z)
```

For `g = λ‖·‖₁` the sub-problem is a quadratic LASSO solved by
coord descent or FISTA. Ideal for logistic-LASSO where the
Hessian is well-conditioned.

## Files

- `python/proximal_newton.py` — Newton IRLS + coord-descent
  LASSO inner solve on logistic-LASSO (n=300, d=30, 5 true
  nonzeros, λ=0.05). Converges to a steady objective ~ 0.52
  in ~5 iterations, illustrating the superlinear rate.
- `r/proximal_newton.R` — `glmnet` (Newton IRLS + coord
  descent), `ncvreg`, `biglasso` (R);
  `sklearn.LogisticRegression`, `celer.LogisticRegression`,
  from-scratch (Python).

## When to use

- **Logistic-LASSO / elastic net** — the archetypal use case
  (glmnet).
- **Composite optimisation** where the Hessian of the smooth
  part is affordable.
- **Sparse GLMs** — Poisson-LASSO, negative-binomial-LASSO,
  Cox-LASSO.

## When NOT to use

- **Very large d** — full Hessian is O(d²); use L-BFGS +
  proximal step or accelerated FISTA.
- **Non-convex penalties** — proximal Newton generalises but
  needs safeguards.
- **When Hessian is expensive to compute** — quasi-Newton /
  FISTA are cheaper.

## Assumptions & caveats

- **Line search / trust region** — needed for global
  convergence; unit step is fine near the optimum.
- **Inner tolerance** — solve the quadratic sub-problem to
  sufficient accuracy (Byrd et al 2016 recommend adaptive
  tolerances).
- **Warm-start** across a λ path — critical for efficiency in
  regularisation paths.
- **Hessian regularisation** — add small λ_H · I when ill-
  conditioned.

## Related in this repo

- `proximal-gradient-method`, `fista-accelerated-proximal`
  — first-order alternatives.
- `lbfgs-quasi-newton` — smooth cousin.
- `safe-screening-lasso` — feature-reduction complement.
- `adaptive-lasso`, `scad-mcp-penalties`, `group-lasso` —
  penalties this framework handles.

## Run

```
python techniques/proximal-newton/python/proximal_newton.py
Rscript techniques/proximal-newton/r/proximal_newton.R
```

**Refs:** Lee, J.D., Sun, Y. and Saunders, M.A. "Proximal Newton-type methods for minimizing composite functions." *SIAM J. Optim.*, 24(3): 1420-1443, 2014.

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
