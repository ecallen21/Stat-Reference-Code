# Gaussian Process Regression (Reference §14.32)

Nonparametric Bayesian regression: place a **Gaussian process** prior over the unknown function `f`, condition on data.

```
f ~ GP(m(x), k(x, x'))
y_i = f(x_i) + ε_i,   ε_i ~ N(0, σ_n²)
```

## Posterior at test points

Given training `(X, y)`, the posterior over `f` at test `X_*` is Gaussian:

```
mean(X_*) = K_{X_*, X} [K_{X, X} + σ_n² I]⁻¹ y
var(X_*)  = K_{X_*, X_*} − K_{X_*, X} [K_{X, X} + σ_n² I]⁻¹ K_{X, X_*}
```

## RBF (squared-exponential) kernel

```
k(x, x') = σ_f² exp(−‖x − x'‖² / (2 ℓ²))
```

Hyperparameters `(ℓ, σ_f, σ_n)` tuned by **maximizing the log marginal likelihood**:

```
log p(y | X) = −½ yᵀ K_y⁻¹ y  −  ½ log|K_y|  −  (n/2) log(2π)
```

## Files

- `python/gaussian_process_regression.py` — from-scratch RBF-kernel GP with L-BFGS marginal-likelihood optimization. Demo (n = 20 draws from `sin(1.5x) + noise 0.15`): fitted `ℓ = 1.20`, `σ_n = 0.089`; interior predictions close to truth with tight variance, extrapolation at `x = ±4` gets wide bands. Matches `sklearn.gaussian_process.GaussianProcessRegressor` to within a few percent.
- `r/gaussian_process_regression.R` — `DiceKriging::km` (canonical R implementation with automatic hyperparameter tuning) or `kernlab::gausspr`.

## Common kernels

- **RBF / squared exponential** — infinitely smooth (default).
- **Matern-3/2, Matern-5/2** — less smooth; better for physical / financial data.
- **Periodic** — for signals with a known period.
- **Sums / products** of kernels — combine trend, seasonality, noise.

## When to use

- **Small-to-moderate `n` (< 5000)** with continuous inputs where an interpretable posterior is more important than raw prediction speed.
- **Emulator** for expensive simulations (aerospace, physics, drug PK).
- **Bayesian optimization surrogate** (see `bayesian-optimization`).
- **Any regression** where calibrated uncertainty matters more than point accuracy.

## Scaling to larger `n`

- **Sparse GP / inducing points** (Titsias 2009): reduce `O(n³)` to `O(n m²)` with `m` inducing points.
- **Nyström** / random Fourier features approximations.
- **Kronecker / Toeplitz** structure for grid inputs.

## Assumptions & caveats

- **Kernel choice** encodes assumed function smoothness; the wrong kernel gives poor extrapolation.
- **Extrapolation** reverts to the prior mean with wide variance — the model correctly flags "I don't know" outside the training range.
- **Multi-output**: extend via coregionalization or independent GPs per output.
- **Cost**: `O(n³)` per fit; `O(n²)` per prediction. Prohibitive for `n > 10⁴` unless sparse.

## Run

```
python techniques/gaussian-process-regression/python/gaussian_process_regression.py
Rscript techniques/gaussian-process-regression/r/gaussian_process_regression.R
```

**Refs:** Rasmussen, C.E. & Williams, C.K.I. *Gaussian Processes for Machine Learning*, MIT Press, 2006; Titsias, M. "Variational learning of inducing variables in sparse Gaussian processes." *AISTATS*, 2009.

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
