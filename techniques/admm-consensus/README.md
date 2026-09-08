# ADMM & Consensus ADMM (Reference §47.68)

Boyd, Parikh, Chu, Peleato & Eckstein (2011). Solve

    min f(x) + g(z)   s.t.  Ax + Bz = c

by alternating

    x^{k+1} = argmin_x  f(x) + (ρ/2)‖Ax + Bz^k − c + u^k‖²
    z^{k+1} = argmin_z  g(z) + (ρ/2)‖Ax^{k+1} + Bz − c + u^k‖²
    u^{k+1} = u^k + Ax^{k+1} + Bz^{k+1} − c.

Ideal for split-friendly objectives (lasso, group lasso, TV) and
distributed / consensus problems (global agreement across agents).

## Files

- `python/admm_consensus.py` — two demos:
  1. **ADMM Lasso** (n=200, p=20, true support {0, 3, 7, 15}):
     recovers the top-4 coefs (1.46, −0.96, 0.73, −0.57) at λ=5.
  2. **Consensus ADMM** — 5 agents each fit a local ridge on 50
     obs of a shared β*; 15 outer iterations recover z within
     0.003 of every coordinate of β*.
- `r/admm_consensus.R` — `ADMM`, `flsa` (R); `cvxpy`,
  `pyproximal`, from-scratch (Python).

## When to use

- **Split-friendly loss + regulariser** — lasso, group lasso, TV,
  sparse GLM, matrix completion.
- **Distributed / federated** learning — consensus ADMM as an
  orchestration primitive.
- **Constrained optimisation** — reformulate constraints as
  augmented-Lagrangian penalties.

## When NOT to use

- **When a good proximal-gradient solver exists** (FISTA / Nesterov)
  — often faster for smooth + prox problems.
- **Highly nonconvex objectives** — no global-optimum guarantees;
  Boyd et al 2011 stresses convex assumption.
- **Tightly-coupled Hessians** — ADMM's block splits don't
  parallelise well.

## Assumptions & caveats

- **Convex f, g** for provable convergence; nonconvex extensions
  work in practice for problems like SDP relaxations.
- **ρ tuning** — over/under-relaxation heuristics (Boyd sec 3.4.1);
  adaptive rho helps.
- **Stopping criteria** — track primal residual r = Ax − Bz + c
  and dual residual s = ρA'B(z − z_prev).
- **Warm starting** across a λ-path halves iteration count.

## Related in this repo

- `coordinate-descent-lasso` — alternative solver for the same
  problem.
- `ridge-lasso-elasticnet`, `group-lasso`, `fused-lasso`,
  `adaptive-lasso`, `scad-mcp-penalties`, `dantzig-selector` —
  regularised regression toolkit.
- `matrix-completion-svt`, `robust-pca` — nuclear-norm cousins
  that ADMM also solves.
- `distributionally-robust-optimization`, `bayesian-optimization` —
  optimisation-heavy neighbours.

## Run

```
python techniques/admm-consensus/python/admm_consensus.py
Rscript techniques/admm-consensus/r/admm_consensus.R
```

**Refs:** Boyd, S., Parikh, N., Chu, E., Peleato, B. & Eckstein, J. "Distributed optimization and statistical learning via the alternating direction method of multipliers." *FnT Machine Learning* 3(1): 1-122, 2011; Parikh, N. & Boyd, S. "Proximal algorithms." *FnT Optim* 1(3): 127-239, 2014.

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
