# Subgradient Method (Reference §47.338)

Shor (1985); Boyd, Xiao & Mutapcic (2004). Minimise a convex
NON-SMOOTH `f` by iterating in ANY subgradient direction:

```
g_k ∈ ∂f(x_k)
x_{k+1} = x_k − α_k · g_k
```

Diminishing step size (`α_k = c / √k`) guarantees convergence
of the RUNNING BEST value at rate `O(1/√k)`; fixed step gives a
neighbourhood.

## Files

- `python/subgradient_method.py` — L1 (LAD) regression with L1
  penalty. n=200, d=10, 2 injected outliers. Running-best
  objective drops from ~ ∞ to ~ 91 by iter 3000; slow but
  correct.
- `r/subgradient_method.R` — `quantreg::rq(tau=0.5)` for LAD
  (usually preferred over subgrad), custom loop; `cvxpy`,
  `statsmodels QuantReg`, from-scratch (Python).

## When to use

- **Non-smooth objectives** without a simple prox — L1 loss
  + L1 penalty, hinge-type composite.
- **Theoretical baseline** for convex non-smooth analysis.
- **Distributed optimisation** — subgradient methods
  parallelise easily (each worker computes a subgradient).

## When NOT to use

- **When a prox operator exists** — use proximal gradient /
  FISTA (much faster).
- **Smooth objectives** — plain gradient descent /
  Nesterov / L-BFGS are all faster.
- **Ill-conditioned problems** — subgradient is very slow;
  bundle methods or ellipsoid may work better.

## Assumptions & caveats

- **Step-size rule** — diminishing `1/√k` gives `O(1/√k)`
  optimality; fixed step gives a neighbourhood of size ~ α G²
  (G = subgradient bound).
- **Running best** — the LAST iterate does NOT converge in
  general; track the best-so-far.
- **Non-differentiable points** — pick any subgradient in
  `∂f`.
- **Projection** — for constrained problems, project after
  each step.

## Related in this repo

- `proximal-gradient-method`, `fista-accelerated-proximal`,
  `proximal-newton` — much faster for structured non-smooth
  problems.
- `admm-consensus` — parallel decomposition alternative.
- `mirror-descent` — geometry-aware cousin.
- `quantile-regression`, `robust-regression` — statistical
  applications.

## Run

```
python techniques/subgradient-method/python/subgradient_method.py
Rscript techniques/subgradient-method/r/subgradient_method.R
```

**Refs:** Shor, N.Z. *Minimization Methods for Non-differentiable Functions*, Springer, 1985; Boyd, S., Xiao, L. and Mutapcic, A. *Subgradient Methods*, Stanford EE392o Lecture Notes, 2004.

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
