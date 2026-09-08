# Hamiltonian Neural Networks (Reference §47.97)

Greydanus, Dzamba & Yosinski (2019). Learn a Hamiltonian H_θ(q, p)
from data, then integrate Hamilton's equations to predict
trajectories:

    dq/dt = +∂H/∂p ,     dp/dt = −∂H/∂q.

Symplectic structure CONSERVES ENERGY (Liouville's theorem),
unlike a plain MLP fitted to (q, p) → (dq/dt, dp/dt) which drifts.

## Files

- `python/hnn_hamiltonian_neural_networks.py` — from-scratch
  HNN with a quadratic + cos(q) basis for H and a standard MLP
  baseline. Demo (pendulum, T=20, dt=0.05, initial energy 0.46):
  - True integration drift = 0.0000
  - **HNN drift = 0.0000** (conserves by construction)
  - MLP drift = 0.30 (loses over half its energy).
- `r/hnn_hamiltonian_neural_networks.R` — no first-class R port;
  `torchdyn`, `hamiltonian-nn` in Python.

## When to use

- **Physical systems** — pendula, orbital mechanics, molecular
  dynamics.
- **Learning conservation laws** from data.
- **Time-reversible simulators** for climate / weather / robotics.
- **Symbolic Hamiltonian discovery** — combine with sparse
  regression (SINDy).

## When NOT to use

- **Dissipative systems** — energy not conserved; use Lagrangian
  NNs with dissipation term or plain MLPs.
- **Non-Hamiltonian dynamics** — check whether a Hamiltonian
  formulation even exists.
- **Very high-dim state** — grad computation quadratic; use
  structured Hamiltonians (Nested-HNN).

## Assumptions & caveats

- **Conservation of H** is exact only with a symplectic integrator
  (e.g. leapfrog, Verlet); RK4 conserves H approximately.
- **Coordinate choice** — canonical (q, p) required.
- **Basis expressivity** — a linear model in fixed basis (as here)
  works when true H is a low-order polynomial + trig; MLPs on H
  needed generally.
- **Noise handling** — HNN's inductive bias regularises noisy
  trajectories.

## Related in this repo

- `neural-ode`, `neural-tangent-kernel` — continuous-time NN
  cousins.
- `hamiltonian-mc`, `hmc-nuts`, `euler-maruyama-sde` — Hamiltonian
  MCMC / SDE integrators.
- `energy-based-models`, `score-matching`,
  `stein-variational-gradient` — energy-function-based methods.
- `state-space-kalman`, `unscented-kalman-filter`,
  `ensemble-kalman-filter` — dynamical-system filters.

## Run

```
python techniques/hnn-hamiltonian-neural-networks/python/hnn_hamiltonian_neural_networks.py
Rscript techniques/hnn-hamiltonian-neural-networks/r/hnn_hamiltonian_neural_networks.R
```

**Refs:** Greydanus, S., Dzamba, M. & Yosinski, J. "Hamiltonian Neural Networks." *NeurIPS*, 2019; Chen, R.T.Q., Rubanova, Y., Bettencourt, J. & Duvenaud, D. "Neural Ordinary Differential Equations." *NeurIPS*, 2018.

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
