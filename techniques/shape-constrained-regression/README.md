# Shape-Constrained Regression (Reference §33.14)

Least-squares regression under a **known shape prior**:

- **Monotone** (nondecreasing) — Pool Adjacent Violators (PAV) algorithm.
- **Convex / concave** — QP with second-difference inequality
  constraints.
- **Unimodal** (single-peak) — PAV on each side of the mode.

Advantages over parametric constrained models:

- **Nonparametric** — no shape family choice.
- **Consistent** under the shape prior even for small n.
- **Interpretable** — the constraint is a scientific prior, not a
  hyperparameter.

## When to use

- **Dose-response, growth, survival curves** — monotonicity is a
  scientific prior.
- **Cost / utility functions** — concavity / convexity is a scientific
  prior.
- **Calibration curves for machine-learning classifiers** — isotonic
  regression is the standard calibrator.

## When NOT to use

- **Shape prior is wrong** — the fit will be biased.
- **Very few observations** — the fit collapses to a step function.
- **Fine derivative estimation** — shape-constrained fits are only
  monotone / convex; derivatives are not smooth.

## Files

- `python/shape_constrained_regression.py` — from-scratch:
  1. `pav(y)` — Pool Adjacent Violators for nondecreasing regression.
  2. `convex_fit(y)` — SLSQP QP with `n-2` second-difference
     inequality constraints.
  Demo on `sqrt(x) + noise`: **PAV MSE 0.0081 → 0.0022**, zero
  monotonicity violations. Demo on `x² + noise`: **convex-QP MSE
  0.0037 → 0.0004**, zero convexity violations.
- `r/shape_constrained_regression.R` — `Iso`, `isotone`, `scam`,
  `cgam`, `cobs` (R); `sklearn.IsotonicRegression`, `cvxpy`,
  `scipy.optimize` (Python).

## Assumptions & caveats

- **Constraint mis-specification bias** — if the truth isn't monotone,
  PAV forces monotonicity and biases toward the closest monotone fit.
- **Discrete step artefacts** — PAV output is piecewise-constant; use
  spline-based constrained methods (`cgam`, `cobs`, `scam`) for smooth
  fits.
- **Multidimensional constraints** — PAV extends to 2-D isotonic on a
  grid; more general shapes require QP.
- **Efficiency** — PAV is `O(n)`; QP for convex is `O(n³)` naive, faster
  with active-set / conic-form solvers.
- **Uncertainty** — bootstrap for pointwise CIs; asymptotics are
  non-Gaussian at the boundary.

## Related in this repo

- `isotonic-regression` — the monotone special case (already in repo).
- `calibration-scaling` — isotonic calibration is a downstream use.
- `additive-quantile-regression`, `gamlss`,
  `distributional-regression` — sibling nonparametric families.
- `restricted-cubic-splines` (if present) — smooth-but-unconstrained
  spline alternatives.

## Run

```
python techniques/shape-constrained-regression/python/shape_constrained_regression.py
Rscript techniques/shape-constrained-regression/r/shape_constrained_regression.R
```

**Refs:** Barlow, R.E., Bartholomew, D.J., Bremner, J.M. & Brunk, H.D. *Statistical Inference under Order Restrictions*, Wiley, 1972; Groeneboom, P. & Jongbloed, G. *Nonparametric Estimation under Shape Constraints*, Cambridge University Press, 2014.

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
