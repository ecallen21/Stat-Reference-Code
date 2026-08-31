# Neural Ordinary Differential Equations — Neural ODE (Reference §27.x extra)

Chen, Rubanova, Bettencourt & Duvenaud (2018). Replace the discrete residual
recurrence:

```
z_{l+1} = z_l + f_θ(z_l)                    (a ResNet block)
```

with a **continuous-depth ODE**:

```
dz/dt = f_θ(z(t), t)
z(t₁) = z(t₀) + ∫_{t₀}^{t₁} f_θ(z(t), t) dt
```

Forward pass = **numerical ODE solver** (Euler, RK4, Dopri5, adaptive).
Depth becomes a hyperparameter of the solver, not a fixed architectural
choice.

## Backprop through the solver

Two options:

- **Direct autodiff** through every solver step. Memory = `O(n_steps · d_hidden)`.
- **Adjoint method** (Pontryagin) — solve a second ODE backward for the gradients. `O(1)` memory but slower per step and sensitive to solver tolerance.

## Applications

- **Continuous normalising flows** (Chen 2018) — invertible-by-construction; density = `log p_z − ∫ tr(∂f/∂z) dt`.
- **FFJORD** (Grathwohl 2018) — Hutchinson trace estimator for scalability.
- **ODE-RNN** (Rubanova 2019) — irregular-time-series modelling.
- **Neural CDE** / **Neural SDE** (Kidger, Morrill 2020) — path-dependent / stochastic extensions.
- **Physics-informed NN** (Raissi 2019) — enforce PDE constraints as regularisers.

## When to use

- **Irregular time series** (medical records, financial ticks) — Neural ODEs handle non-uniform time stamps naturally.
- **Density estimation** with exact-likelihood continuous normalising flows.
- **Physics-aware modelling** where the vector field has structural meaning.
- **Adjoint** memory savings when depth is huge or memory is tight.
- **NOT** for straightforward classification / regression — discrete networks are cheaper and just as accurate.

## Files

- `python/neural_ode.py` — from-scratch Neural ODE with a tiny MLP `f_θ(z, t)`, Euler and RK4 solvers. Convergence demo: Euler `n_steps = 5 / 20 / 200`, RK4 `n_steps = 20 / 200`. RK4 with 20 steps matches RK4-200 to 2e-9; Euler-200 vs RK4-200 error 9e-4 — the standard order-4 vs order-1 improvement.
- `r/neural_ode.R` — `deSolve::ode` for classical ODEs; `reticulate` + `torchdiffeq` (PyTorch), `diffrax` (JAX).

## Assumptions & caveats

- **Solver tolerance is a hyperparameter** — too loose gives noisy gradients; too tight is slow.
- **Vector-field expressivity** — depth-of-integration determines what the field can compute; short integration intervals need very expressive `f_θ`.
- **Stability** — `f_θ` should be Lipschitz for numerical stability; regularise its gradient norm.
- **Adjoint method** — memory-efficient but backward errors can accumulate; use in production with care.
- **Batched integration** — different batch elements may need different `t` grids for irregular time series.
- **Continuous-depth ≠ better** — on standard image / language benchmarks discrete transformers still win; NODEs shine on physics / time-series with structural priors.

## Related in this repo

- `residual-connections` — the discrete analogue; Euler with 1 step per block.
- `state-space-models` — related linear-dynamical-system primitive.
- `normalizing-flows` — CNF is a Neural ODE with tractable log-det.
- `diffusion-model` — score-based diffusion has a Neural-ODE interpretation of its probability-flow reverse process.

## Run

```
python techniques/neural-ode/python/neural_ode.py
Rscript techniques/neural-ode/r/neural_ode.R
```

**Refs:** Chen, R.T.Q. et al. "Neural ordinary differential equations." *NeurIPS*, 2018; Grathwohl, W. et al. "FFJORD: free-form continuous dynamics for scalable reversible generative models." *ICLR*, 2019; Rubanova, Y., Chen, R.T.Q. & Duvenaud, D. "Latent ODEs for irregularly-sampled time series." *NeurIPS*, 2019; Kidger, P. *On Neural Differential Equations*, PhD thesis, Oxford, 2021.

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
