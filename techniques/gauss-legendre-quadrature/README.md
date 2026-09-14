# Gauss-Legendre Quadrature (Reference §47.363)

Gauss (1814). Approximate `∫_{−1}^{1} f(x) dx` by

```
Σ_{i=1}^{n} w_i · f(x_i)
```

where `{x_i}` are the roots of the n-th Legendre polynomial
and `{w_i}` are the associated weights (Golub-Welsch
algorithm: eigenvalues of a symmetric tridiagonal Jacobi
matrix). n-point Gauss-Legendre is EXACT for polynomials of
degree `≤ 2n − 1` — twice the degree of Newton-Cotes with the
same number of nodes.

## Files

- `python/gauss_legendre_quadrature.py` — Four test
  integrands. GL nails polynomials to machine precision at
  n=8; on Gaussian and oscillatory integrands GL beats
  trapezoidal / Simpson by 3-6 orders of magnitude at n=16.
- `r/gauss_legendre_quadrature.R` —
  `statmod::gauss.quad(kind='legendre')`,
  `pracma::gaussLegendre`, `stats::integrate` (R);
  `numpy.polynomial.legendre.leggauss`,
  `scipy.integrate.fixed_quad`, from-scratch (Python).

## When to use

- **Smooth integrands** — analytic on `[a, b]` gives
  exponential convergence.
- **Bayesian marginal likelihood** — quadrature-based marginal
  in low-D hyperparameter integrals.
- **GLMM adaptive Gauss-Hermite** — random-effect integrals
  in `glmer` / `lme4` / `PROC NLMIXED`.
- **Finite-element mass matrices** — closed-form when
  polynomials, Gauss-Legendre otherwise.

## When NOT to use

- **Singular / oscillatory integrands** — use Clenshaw-Curtis,
  Gauss-Chebyshev (weight `1/√(1-x²)`), or specialised
  Filon / Levin methods.
- **High dimension** — quadrature scales as `n^d`; use
  quasi-Monte Carlo / sparse-grid methods.
- **Improper integrals** — use Gauss-Laguerre / Gauss-Hermite
  for infinite domains.

## Assumptions & caveats

- **Node computation** — Golub-Welsch: eigendecomp of the
  Jacobi matrix (O(n²)); or fast recursive methods
  (O(n) node-by-node).
- **Interval mapping** — for `[a, b]` use
  `t = (b−a)/2 · x + (a+b)/2`, weights scale by `(b−a)/2`.
- **Related weight families** — Chebyshev (1st/2nd kind),
  Hermite, Laguerre, Jacobi; all have Golub-Welsch
  eigenvalue implementations.
- **Adaptive Gauss-Kronrod** — Kronrod adds n+1 nodes that
  reuse the Gauss nodes for error estimation; QUADPACK
  default.

## Related in this repo

- `runge-kutta-integration` — ODE integration (a related
  numerical-analysis pillar).
- `monte-carlo-simulation`, `quasi-monte-carlo-sobol`,
  `latin-hypercube-sampling` — random / low-discrepancy
  alternatives.
- `chebyshev-approximation` — orthogonal-polynomial cousin
  used for spectral methods.
- `hmc-nuts`, `variational-inference` — Bayesian integration
  alternatives.

## Run

```
python techniques/gauss-legendre-quadrature/python/gauss_legendre_quadrature.py
Rscript techniques/gauss-legendre-quadrature/r/gauss_legendre_quadrature.R
```

**Refs:** Gauss, C.F. "Methodus nova integralium valores per approximationem inveniendi." *Comm. Soc. Regiae Sci. Gottingensis*, 3, 1814; Golub, G.H. and Welsch, J.H. "Calculation of Gauss quadrature rules." *Math. Comput.*, 23: 221-230, 1969.

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
