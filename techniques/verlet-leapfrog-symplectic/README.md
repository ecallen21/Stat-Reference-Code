# Velocity Verlet / Leapfrog Symplectic Integrator (Reference §47.370)

Verlet (1967); Hairer-Lubich-Wanner (2006, *Geometric
Numerical Integration*). For a separable Hamiltonian
`H(q, p) = ½ pᵀ M⁻¹ p + V(q)` do a half-kick / full-drift /
half-kick:

```
p_{k+½} = p_k − ½ h · ∇V(q_k)
q_{k+1} = q_k + h · M⁻¹ p_{k+½}
p_{k+1} = p_{k+½} − ½ h · ∇V(q_{k+1})
```

SYMPLECTIC — preserves phase-space volume; energy oscillates
around the exact value but does NOT drift. Backbone of
molecular dynamics, N-body simulation, and HMC / NUTS.

## Files

- `python/verlet_leapfrog_symplectic.py` — Harmonic
  oscillator `H = ½(p² + q²)` for T=5 000 time units at
  h=0.5. Verlet's energy stays bounded within `−3.1e−2`
  of exact 0.5; RK4's energy DECAYS from 0.5 to 0.061 (a
  ~ 88% loss) because it is not symplectic. Illustrates
  the long-time drift phenomenon.
- `r/verlet_leapfrog_symplectic.R` — `deSolve` (limited),
  custom loop (R); `blackjax` HMC (leapfrog internal),
  `astropy.integrate.leapfrog`, from-scratch (Python).

## When to use

- **Long-time simulation of Hamiltonian systems** — MD, N-
  body, orbital mechanics.
- **HMC / NUTS proposals** — Neal 2011; the reason Verlet is
  the DEFAULT leapfrog in every Stan / PyMC / blackjax HMC
  implementation.
- **Any conservative system where energy drift is a bug** —
  climate coupled systems, plasma physics.

## When NOT to use

- **Non-Hamiltonian / dissipative systems** — the symplectic
  property does nothing.
- **Stiff Hamiltonian systems** — implicit symplectic (Gauss-
  Legendre) needed; explicit leapfrog goes unstable.
- **When higher order is important** — Yoshida-4 or Forest-
  Ruth composition improves accuracy; leapfrog is only 2nd
  order.

## Assumptions & caveats

- **Separable Hamiltonian** — `H = T(p) + V(q)`; general H
  requires implicit symplectic (Gauss-Legendre) or splitting.
- **Time-reversibility** — leapfrog is exactly time-
  reversible; important for HMC detailed balance.
- **Time-step stability** — for the harmonic oscillator
  leapfrog is stable for `|hω| < 2`; larger h ⇒ instability.
- **Yoshida-4 / composition** — apply leapfrog three times
  with negative middle step for 4th-order accuracy.
- **Volume-preservation** — Jacobian of the leapfrog map is
  1; this is why HMC uses it (no correction term).

## Related in this repo

- `hmc-nuts`, `hamiltonian-mc`, `mala-langevin` — MCMC that
  relies on the symplectic property.
- `runge-kutta-integration` — non-symplectic ODE cousin.
- `neural-ode`, `hnn-hamiltonian-neural-networks` — modern
  neural analogues.
- `euler-maruyama-sde`, `stochastic-gradient-mcmc` —
  stochastic dynamics siblings.

## Run

```
python techniques/verlet-leapfrog-symplectic/python/verlet_leapfrog_symplectic.py
Rscript techniques/verlet-leapfrog-symplectic/r/verlet_leapfrog_symplectic.R
```

**Refs:** Verlet, L. "Computer 'experiments' on classical fluids. I. Thermodynamical properties of Lennard-Jones molecules." *Phys. Rev.*, 159: 98-103, 1967; Hairer, E., Lubich, C. and Wanner, G. *Geometric Numerical Integration*, 2nd ed., Springer, 2006.

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
