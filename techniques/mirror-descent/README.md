# Mirror Descent (Reference §47.339)

Nemirovsky & Yudin (1983); Beck & Teboulle (2003). Generalise
gradient descent by choosing a MIRROR MAP `Φ` (Bregman
generator):

```
x_{k+1} = argmin_x ⟨∇f(x_k), x⟩ + (1/η) D_Φ(x, x_k)
```

For probability-simplex constraints, `Φ(x) = Σ x_i log x_i`
recovers EXPONENTIATED GRADIENT (multiplicative weights):

```
x_{k+1, j} ∝ x_{k, j} · exp(−η ∇f_j)
```

Adapts step geometry to the constraint set, often beating plain
projected gradient in high-dimensional simplex settings.

## Files

- `python/mirror_descent.py` — Exponentiated gradient
  (Bregman on the simplex with entropy generator) vs Duchi
  et al 2008 projected gradient. Convex objective on the
  d=20 simplex. Both find the same active set; exp-grad
  converges smoothly without projection.
- `r/mirror_descent.R` — base R custom loop (R); custom
  `torch` autograd + softmax step, `cvxpy`, from-scratch
  (Python).

## When to use

- **Probability-simplex constraints** (portfolio weights,
  categorical distributions) — exp-grad natural.
- **Very high-dimensional simplex** — Euclidean projection is
  expensive.
- **Online learning** — multiplicative-weights / experts /
  hedge algorithms are mirror descent in disguise.

## When NOT to use

- **When your constraint set has no matching Bregman map** —
  fall back to projected gradient.
- **Smooth unconstrained problems** — first-order methods are
  simpler.
- **Very small d** — projected gradient's cost is trivial.

## Assumptions & caveats

- **Convexity of `Φ`** and dual feasibility of `∇Φ*` — needed
  for a well-defined update.
- **Step-size η** — theory: `O(√(D_max / T))`; practice: line
  search or diminishing schedule.
- **Normalisation** — exp-grad update must renormalise; log-
  space arithmetic prevents overflow.
- **Convergence rate** — `O(√(log d / T))` on the simplex vs
  Euclidean's `O(√(d / T))` — huge gap in high dim.

## Related in this repo

- `proximal-gradient-method`, `fista-accelerated-proximal`
  — proximal alternatives.
- `subgradient-method` — general non-smooth alternative.
- `exponentiated-gradient-reduction` — a specific
  fairness-constrained variant.
- `multi-armed-bandits`, `thompson-sampling` — related
  online-learning framework.

## Run

```
python techniques/mirror-descent/python/mirror_descent.py
Rscript techniques/mirror-descent/r/mirror_descent.R
```

**Refs:** Nemirovsky, A.S. and Yudin, D.B. *Problem Complexity and Method Efficiency in Optimization*, Wiley, 1983; Beck, A. and Teboulle, M. "Mirror descent and nonlinear projected subgradient methods for convex optimization." *Oper. Res. Lett.*, 31(3): 167-175, 2003.

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
