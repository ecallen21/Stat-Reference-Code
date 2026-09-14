# Chebyshev Approximation (Reference §47.364)

Chebyshev (1854); Clenshaw-Curtis (1960); Trefethen (2013).
For smooth `f` on `[−1, 1]` the Chebyshev series

```
f(x) ≈ Σ_{k=0}^{N} c_k · T_k(x)
```

converges GEOMETRICALLY (Bernstein ellipse rate). Coefficients
via a DCT-II applied to `f` sampled at the Chebyshev-Gauss-
Lobatto nodes `x_k = cos(kπ/N)`. Enables:

- **Near-minimax uniform approximation** — sup-norm error
  close to the theoretical best polynomial.
- **Clenshaw-Curtis quadrature** — closed-form weights,
  competitive with Gauss-Legendre.
- **Barycentric interpolation** — numerically stable
  evaluation at arbitrary points.
- **Chebyshev spectral methods** — for PDEs on tensor-product
  domains.

## Files

- `python/chebyshev_approximation.py` — Four test functions.
  Smooth `exp(x)` hits machine precision at N=20;
  `sin(4πx)` hits it at N=40; Runge's `1/(1+25x²)` decays
  geometrically to 1e-7 at N=80; non-smooth `|x|` only
  `O(1/N)` convergence — exactly the theoretical picture.
- `r/chebyshev_approximation.R` — `chebpol::ipol(method='chebyshev')`,
  `pracma::chebPoly`/`chebApprox` (R);
  `numpy.polynomial.chebyshev.Chebyshev.fit`, `chebpy`,
  from-scratch (Python).

## When to use

- **Function approximation on a compact interval** where
  smoothness is available.
- **Spectral solvers** for smooth PDEs (radiation, elasticity,
  fluid dynamics).
- **Numerical differentiation / integration** by
  differentiating / integrating the polynomial.
- **Rational surrogate models** — combined with
  Padé-Chebyshev approximants.

## When NOT to use

- **Non-smooth or oscillatory functions** — geometric
  convergence breaks down; use spline / wavelets /
  matching-pursuit.
- **High dimension** — Chebyshev tensor product grows
  exponentially; use sparse-grid Smolyak or Monte-Carlo.
- **Fast periodic problems** — Fourier series are cleaner
  (Chebyshev is Fourier after `x = cos θ`).

## Assumptions & caveats

- **Domain [−1, 1]** — map `t = (b − a)/2 · x + (a + b)/2`
  for `[a, b]`.
- **Node sampling** — Chebyshev-Gauss-Lobatto nodes cluster
  at endpoints, defeating the Runge phenomenon that hits
  equispaced Lagrange interpolation.
- **Clenshaw's recurrence** — numerically stable evaluation
  of the sum, vs the direct sum which loses precision for
  large N.
- **Coefficient decay** — for analytic `f`, coefficients decay
  `|c_k| ~ ρ^{−k}` where `ρ > 1` is the Bernstein-ellipse
  parameter.

## Related in this repo

- `gauss-legendre-quadrature` — Clenshaw-Curtis is its main
  competitor for smooth integrands.
- `splines-regression`, `thin-plate-splines`,
  `functional-basis-smoothing` — competing bases.
- `nonlinear-least-squares`, `gam` — non-parametric
  regression using Chebyshev basis.
- `neural-ode`, `runge-kutta-integration` — numerical-
  analysis siblings.

## Run

```
python techniques/chebyshev-approximation/python/chebyshev_approximation.py
Rscript techniques/chebyshev-approximation/r/chebyshev_approximation.R
```

**Refs:** Chebyshev, P.L. "Théorie des mécanismes connus sous le nom de parallélogrammes." *Mém. Prés. Acad. Sci. St.-Petersburg*, 7: 539-568, 1854; Clenshaw, C.W. and Curtis, A.R. "A method for numerical integration on an automatic computer." *Numer. Math.*, 2: 197-205, 1960; Trefethen, L.N. *Approximation Theory and Approximation Practice*, SIAM, 2013.

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
