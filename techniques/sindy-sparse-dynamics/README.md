# SINDy - Sparse Identification of Nonlinear Dynamics (Ref §47.125)

Brunton, Proctor & Kutz (2016). Given time-series `x(t)` and its
derivatives `dx/dt`, fit each equation as a SPARSE linear
combination of candidate basis functions `Θ(x)` (polynomials, trig,
etc.):

    dX/dt = Θ(X) · Ξ,     Ξ sparse.

Sequentially thresholded least squares (STLSQ) alternately fits
active coefficients by least squares and zeroes out ones with
|Ξ_ij| < λ.

## Files

- `python/sindy_sparse_dynamics.py` — from-scratch polynomial
  basis + STLSQ. Demo (Van der Pol oscillator, RK4 simulator,
  central-difference derivatives, cubic polynomial basis):
  - Recovered: `dx0/dt = +1.000·x1`,
    `dx1/dt = −1.000·x0 + 1.000·x1 − 1.000·x0²·x1`
  - Truth:    `dx0/dt = y`,
    `dy/dt = μ(1 − x²)y − x = y − x²y − x` with μ=1.
  Exact recovery.
- `r/sindy_sparse_dynamics.R` — no first-class R port; `pysindy`
  (Python), `sindyr` (R), `torchdyn` for GPU.

## When to use

- **Physics discovery** — identify governing equations from data.
- **Reduced-order modelling** — sparse-basis surrogate of a heavy
  simulator.
- **PDE discovery** — SINDy-PI, SINDy-PDE extensions.
- **Control-oriented modelling** — sparse models identify quickly.

## When NOT to use

- **Unknown basis** — SINDy requires you to specify the candidate
  library.
- **Very noisy derivatives** — Total-Variation or Savitzky-Golay
  smoothing first.
- **Stochastic dynamics** — extend to SINDy for SDEs (Boninsegna 2018).
- **Highly-nonlinear scaling** — the library grows quickly with
  polynomial order.

## Assumptions & caveats

- **Derivative estimation** — the biggest error source; central
  finite differences require dense sampling.
- **Basis choice** — polynomials + trig cover many physics ODEs;
  neural-net libraries (SINDy-AE) generalise.
- **λ tuning** — cross-validate or use Pareto-front analysis
  (accuracy vs sparsity).
- **Identifiability**: correlated basis functions produce
  ambiguous Ξ; consider PDE-FIND / weak-form SINDy.

## Related in this repo

- `hnn-hamiltonian-neural-networks`, `neural-ode`,
  `dynamic-mode-decomposition-dmd` — data-driven dynamics
  cousins.
- `coordinate-descent-lasso`, `adaptive-lasso`, `group-lasso`,
  `scad-mcp-penalties`, `debiased-lasso`,
  `orthogonal-matching-pursuit`, `lars-least-angle-regression` —
  sparse regression toolbox.
- `wavelet-analysis`, `empirical-mode-decomposition`,
  `welch-power-spectral-density` — signal-analysis cousins.

## Run

```
python techniques/sindy-sparse-dynamics/python/sindy_sparse_dynamics.py
Rscript techniques/sindy-sparse-dynamics/r/sindy_sparse_dynamics.R
```

**Refs:** Brunton, S.L., Proctor, J.L. & Kutz, J.N. "Discovering governing equations from data by sparse identification of nonlinear dynamical systems." *PNAS* 113(15): 3932-3937, 2016.

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
