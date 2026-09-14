# Chambolle-Pock Primal-Dual (Reference §47.381)

Chambolle & Pock (2011). Solve saddle-point problems of the
form `min_x max_y ⟨Kx, y⟩ + f(x) − g*(y)`, which arise from
composite objectives `min_x f(x) + g(Kx)` via Fenchel duality:

```
y_{k+1} = prox_{σ g*}(y_k + σ K x̄_k)
x_{k+1} = prox_{τ f}(x_k − τ Kᵀ y_{k+1})
x̄_{k+1} = x_{k+1} + θ (x_{k+1} − x_k)     (extrapolation)
```

Convergence when `τ · σ · ‖K‖² ≤ 1`. Ideal for TV denoising
and imaging inverse problems where `K` is a difference or
sparse-linear operator. Faster than Douglas-Rachford when the
linear operator is available in a matrix-free way.

## Files

- `python/chambolle_pock_primal_dual.py` — 1D TV denoising
  on a piecewise-constant signal (n=100, noise σ=0.3, λ=0.5).
  Noisy-signal RMSE 0.31 drops to 0.12 after 500 iterations;
  the recovered signal has 82 near-flat plateau segments,
  consistent with piecewise-constant recovery.
- `r/chambolle_pock_primal_dual.R` — `flare`, `imager` for
  2D image (R); `scikit-image.restoration.denoise_tv_chambolle`,
  `pyproximal.primaldual`, from-scratch (Python).

## When to use

- **TV / anisotropic-diffusion image denoising** — CP is the
  reference algorithm.
- **Linear inverse problems with sparse regularisation** —
  image deblurring, MRI reconstruction, compressed sensing.
- **Composite non-smooth problems with a linear operator K**
  — when `K` is difference / gradient / graph-Laplacian.

## When NOT to use

- **When K is identity** — Douglas-Rachford or FISTA is
  simpler.
- **Smooth-only problems** — L-BFGS is faster.
- **When PROX of `g*` (Fenchel conjugate) is hard** — some
  regularisers don't have closed-form dual prox.

## Assumptions & caveats

- **Step-size product** — `τ σ ‖K‖² < 1` is required;
  adaptive variants (Chambolle-Pock-2 with strong-convex
  acceleration) exist.
- **Operator norm ‖K‖** — for gradient `K` in image
  processing, `‖K‖² ≤ 4·dim` (dim = 1 or 2).
- **θ extrapolation** — `θ = 1` is standard;
  Nesterov-style acceleration when `f` or `g*` is strongly
  convex uses adaptive θ.
- **Convergence rate** — `O(1/k)` in ergodic sense; linear
  under strong convexity.
- **Warm-start** — CP benefits from initialising close to
  the solution; `x₀ = noisy` a common choice.

## Related in this repo

- `douglas-rachford-splitting`, `admm-consensus`,
  `proximal-gradient-method`, `fista-accelerated-proximal` —
  splitting-method neighbours.
- `total-variation-denoising` (if present), `robust-pca`,
  `fused-lasso`, `wavelet-analysis` — regularised inverse
  problems the CP algorithm handles.
- `ridge-lasso-elasticnet`, `group-lasso` — statistical
  applications.

## Run

```
python techniques/chambolle-pock-primal-dual/python/chambolle_pock_primal_dual.py
Rscript techniques/chambolle-pock-primal-dual/r/chambolle_pock_primal_dual.R
```

**Refs:** Chambolle, A. and Pock, T. "A first-order primal-dual algorithm for convex problems with applications to imaging." *J. Math. Imaging Vision*, 40(1): 120-145, 2011.

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
