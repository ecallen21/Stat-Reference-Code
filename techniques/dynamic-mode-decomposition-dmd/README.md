# Dynamic Mode Decomposition (Reference §47.126)

Schmid (2010). Data-driven approximation of the Koopman operator.
Given snapshots `X = [x₀, …, x_{n−1}]`, `X' = [x₁, …, x_n]`, find
low-rank `A` minimising `‖X' − AX‖_F`. Exact-DMD (Tu et al 2014):

    U, S, V = SVD(X, r)
    Ã = Uᴴ X' V S⁻¹
    Φ = X' V S⁻¹ W    (W: eigvecs of Ã, eigvals Λ)
    x(t) = Φ exp(Ω t) b,   Ω = log Λ / dt.

Predicts future snapshots and extracts spatiotemporal MODES.

## Files

- `python/dynamic_mode_decomposition_dmd.py` — from-scratch
  Exact-DMD. Demo (two spatial modes × two complex frequencies,
  rank 4):
  - dominant angular frequencies recovered = [2.0, 3.0] (truth)
  - relative reconstruction error = 0.0000.
- `r/dynamic_mode_decomposition_dmd.R` — no first-class R port;
  `pydmd` (Python), `dyn.dmd` (Julia).

## When to use

- **Fluid dynamics** — extract coherent structures (Schmid original).
- **Video / high-dim time series** — data-driven Koopman modes.
- **Short-term forecasting** — DMD gives explicit exp(Ωt) form.
- **Control** — DMD with control (DMDc) for actuated systems.

## When NOT to use

- **Strong nonlinearity** — Extended DMD or Koopman-with-neural-
  lifts.
- **Noisy data** — use tls-DMD or Total-Least-Squares DMD.
- **Very long-horizon prediction** — Ω sensitive; error grows.
- **Streaming** — use online DMD variants.

## Assumptions & caveats

- **Rank r** must be chosen — usually via singular-value elbow.
- **Complex eigenvalues** produce oscillatory modes; take
  real projections for interpretation.
- **Stability** — `|Λ| ≤ 1` for undamped modes; > 1 diverges.
- **Snapshots must be equally spaced in time**.

## Related in this repo

- `sindy-sparse-dynamics`, `neural-ode`,
  `hnn-hamiltonian-neural-networks` — data-driven-dynamics
  neighbours.
- `functional-pca`, `probabilistic-pca`, `sparse-pca`,
  `matrix-completion-svt`, `robust-pca` — SVD-based cousins.
- `state-space-kalman`, `state-space-models`,
  `particle-filter-smc`, `rts-kalman-smoother` — state-space
  alternatives.
- `wavelet-analysis`, `welch-power-spectral-density`,
  `empirical-mode-decomposition` — signal-analysis cousins.

## Run

```
python techniques/dynamic-mode-decomposition-dmd/python/dynamic_mode_decomposition_dmd.py
Rscript techniques/dynamic-mode-decomposition-dmd/r/dynamic_mode_decomposition_dmd.R
```

**Refs:** Schmid, P.J. "Dynamic mode decomposition of numerical and experimental data." *J Fluid Mech* 656: 5-28, 2010; Tu, J.H., Rowley, C.W., Luchtenburg, D.M., Brunton, S.L. & Kutz, J.N. "On dynamic mode decomposition: theory and applications." *J Comp Dyn* 1(2): 391-421, 2014.

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
