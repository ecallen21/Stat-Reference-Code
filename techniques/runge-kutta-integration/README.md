# Runge-Kutta 4 (Reference §47.362)

Runge (1895); Kutta (1901). Classical explicit 4-stage 4th-
order integrator for `dy/dt = f(t, y)`:

```
k₁ = f(t, y)
k₂ = f(t + h/2, y + h/2 · k₁)
k₃ = f(t + h/2, y + h/2 · k₂)
k₄ = f(t + h,   y + h · k₃)
y_new = y + h/6 · (k₁ + 2k₂ + 2k₃ + k₄)
```

Local truncation error `O(h⁵)`, global `O(h⁴)`. Backbone of
`scipy.integrate.solve_ivp` (default RK45 = Dormand-Prince),
`deSolve::ode`, and every intro ODE textbook.

## Files

- `python/runge_kutta_integration.py` — Harmonic oscillator
  y'' + y = 0 to t=10. Errors converge with `O(h⁴)` for RK4:
  `h=0.1 → 7e−6, h=0.05 → 4e−7, h=0.01 → 7e−10`. Euler
  errors are 4-5 orders of magnitude larger and only `O(h)`.
  Also runs the Lorenz-63 chaotic system.
- `r/runge_kutta_integration.R` — `deSolve::ode(method='rk4')`
  (Soetaert), `pracma::rk4` (R);
  `scipy.integrate.solve_ivp(method='RK45')`, `torchdiffeq`,
  from-scratch (Python).

## When to use

- **Non-stiff smooth ODEs** with moderate accuracy needs
  (physics, chemical kinetics, population dynamics).
- **Chaotic systems** — Lorenz, Rossler; RK4 conserves
  qualitative behaviour longer than Euler.
- **Baseline** for adaptive schemes (RK45, RKF).
- **Neural ODEs** — the `torchdiffeq` package uses RK-family
  integrators with adjoint-differentiation.

## When NOT to use

- **Stiff ODEs** — chemical kinetics with fast/slow scales;
  use BDF (backward differentiation) / implicit Runge-Kutta.
- **When exact energy conservation matters** — Hamiltonian
  systems need symplectic integrators (leapfrog / Verlet).
- **Very long integration** — errors accumulate; use
  Richardson extrapolation, symplectic, or Gauss-Legendre
  implicit RK.

## Assumptions & caveats

- **Step-size control** — fixed-step RK4 lacks it; use
  Dormand-Prince (RK45, RKF) or Cash-Karp for adaptive.
- **Stability region** — RK4 is unstable for `|hλ| > ~ 2.78`
  on the imaginary axis and about the same on the real
  negative axis; needs small h for stiff problems.
- **Order barriers** — Butcher's order-4 barrier: 4-stage
  explicit RK cannot exceed order 4; higher orders need more
  stages.
- **Symplecticity** — RK4 is NOT symplectic; long-term
  energy drift for Hamiltonians.

## Related in this repo

- `neural-ode`, `hnn-hamiltonian-neural-networks` — ODE-based
  neural nets.
- `state-space-kalman`, `extended-kalman-filter` — continuous-
  time dynamics linearised.
- `euler-maruyama-sde`, `score-based-sde` — stochastic
  neighbours.
- `gauss-legendre-quadrature`, `chebyshev-approximation` —
  numerical-analysis siblings.

## Run

```
python techniques/runge-kutta-integration/python/runge_kutta_integration.py
Rscript techniques/runge-kutta-integration/r/runge_kutta_integration.R
```

**Refs:** Runge, C. "Ueber die numerische Auflösung von Differentialgleichungen." *Math. Ann.*, 46: 167-178, 1895; Kutta, W. "Beitrag zur näherungsweisen Integration totaler Differentialgleichungen." *Z. Math. Phys.*, 46: 435-453, 1901.

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
