# MALA — Metropolis-Adjusted Langevin (Reference §47.347)

Roberts & Tweedie (1996). Overdamped Langevin dynamics
discretised by Euler-Maruyama produces the proposal:

```
y = x + τ ∇log π(x) + √(2τ) η,   η ~ N(0, I)
```

Metropolis-Hastings accepts with
`α = min(1, π(y) q(x|y) / (π(x) q(y|x)))`, correcting the
discretisation bias of unadjusted Langevin (ULA). Roberts &
Rosenthal (1998) showed the optimal acceptance rate is
**0.574** on isotropic Gaussians as `d → ∞`.

## Files

- `python/mala_langevin.py` — Correlated 5-D Gaussian target
  with 20 000 draws (first 2 000 discarded). At τ = 1.0
  accept-rate ≈ 0.73 (close to optimum 0.574) and empirical
  covariance matches the target Σ to Frobenius error ~ 3 %;
  τ = 2.0 shows overshoot (rate 0.37, error up again).
- `r/mala_langevin.R` — `LaplacesDemon(Algorithm='MALA')`,
  `mcmc::mcmc.metrop` (R); `blackjax.mala`, `pyro.MALA`,
  from-scratch (Python).

## When to use

- **Continuous targets with cheap gradient** — Bayesian
  posteriors of GLM, hierarchical model, Gaussian process.
- **Higher effective sample size per second** than RW
  Metropolis in medium dim.
- **Autodiff-friendly workflows** — plug in JAX / PyTorch.

## When NOT to use

- **Discrete or heavy-tailed targets** — MALA can miss the
  mode; use RWM, HMC, or specialised methods.
- **Very ill-conditioned targets** — precondition with
  Riemannian metric (Girolami-Calderhead 2011) or use HMC.
- **Multi-modal targets** — replica exchange or annealing
  needed.

## Assumptions & caveats

- **Step-size τ** — adaptive `τ ∝ d^(−1/3)` (Roberts-
  Rosenthal 1998).
- **Preconditioning** — for correlated Σ, use
  `τ Σ_prop ∇log π + √(2 τ Σ_prop) η` with `Σ_prop ≈ Σ`.
- **Warm-up** — needs a few hundred to a few thousand steps
  to forget the initial state.
- **Detailed balance** — proposal density q is Gaussian
  centred at the drifted point; must be evaluated correctly.

## Related in this repo

- `hmc-nuts`, `hamiltonian-mc` — gradient-informed but
  symplectic; often preferred in higher dim.
- `mcmc-metropolis-hastings`, `adaptive-metropolis-haario`
  — non-gradient MCMC baselines.
- `stochastic-gradient-mcmc` — minibatch cousin (SGLD).
- `stein-variational-gradient` — deterministic particle
  alternative.

## Run

```
python techniques/mala-langevin/python/mala_langevin.py
Rscript techniques/mala-langevin/r/mala_langevin.R
```

**Refs:** Roberts, G.O. and Tweedie, R.L. "Exponential convergence of Langevin distributions and their discrete approximations." *Bernoulli*, 2: 341-363, 1996; Roberts, G.O. and Rosenthal, J.S. "Optimal scaling of discrete approximations to Langevin diffusions." *JRSS-B*, 60(1): 255-268, 1998.

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
