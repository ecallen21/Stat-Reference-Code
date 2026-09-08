# Nesterov Accelerated Gradient (Reference §47.107)

Nesterov (1983). For a smooth convex `f` with L-Lipschitz gradient,
plain gradient descent converges O(1/k); NAG achieves O(1/k²) via
a LOOK-AHEAD gradient:

    y_k = θ_k + β_k (θ_k − θ_{k−1})
    θ_{k+1} = y_k − (1/L) ∇f(y_k).

FISTA (Beck & Teboulle 2009) schedules β_k via
`t_{k+1} = (1 + √(1 + 4 t_k²)) / 2`, β_k = (t_{k−1} − 1)/t_k.

## Files

- `python/nesterov_accelerated_gradient.py` — from-scratch GD /
  Heavy-Ball / NAG on the ill-conditioned quadratic
  `f(x) = 0.5 xᵀ diag(1, 100) x`. Demo (κ=100, lr=1/L):
  - GD:         final ‖x‖ = 1.34; >200 iters to hit 1e-3
  - Heavy-Ball: final ‖x‖ = 5.2e-2
  - NAG:        final ‖x‖ = 3.7e-2; **162 iters to 1e-3**.
- `r/nesterov_accelerated_gradient.R` — `torch::optim_sgd(
  momentum, nesterov=TRUE)` (R); `torch.optim.SGD(nesterov=True)`
  in Python.

## When to use

- **Smooth convex optimisation** with known L.
- **Lasso / ridge** solvers — FISTA is the proximal-gradient version.
- **Deep learning** — SGD-momentum-Nesterov is a standard choice.
- **Any first-order method** where you want a rate speedup at
  minimal extra cost.

## When NOT to use

- **Non-smooth non-proximable losses** — use subgradient / bundle
  methods.
- **Strongly convex, small κ** — Heavy-Ball can be marginally faster.
- **Very stochastic gradients** — noise breaks the acceleration
  guarantees; Adam / RMSprop more forgiving.

## Assumptions & caveats

- **Lipschitz constant L** — needed for step size; back-track if
  unknown (FISTA-BT).
- **Momentum schedule** matters — heavy-ball momentum needs
  tuning; FISTA schedule adapts.
- **Restarts** (O'Donoghue-Candès 2015) improve behaviour when
  objective is only "close to" convex.
- **Non-convex** deep-learning practice: constant momentum
  β = 0.9 usually good enough.

## Related in this repo

- `rmsprop-optimizer`, `adam-optimizer`, `online-learning-sgd`,
  `lr-schedules`, `gradient-clipping` — optimisation toolbox.
- `coordinate-descent-lasso`, `admm-consensus`,
  `frank-wolfe-conditional-gradient` — convex-optim family.
- `distributionally-robust-optimization`, `bayesian-optimization`
  — constrained / global optimisation neighbours.

## Run

```
python techniques/nesterov-accelerated-gradient/python/nesterov_accelerated_gradient.py
Rscript techniques/nesterov-accelerated-gradient/r/nesterov_accelerated_gradient.R
```

**Refs:** Nesterov, Y. "A method for solving the convex programming problem with convergence rate O(1/k²)." *Dokl Akad Nauk SSSR* 269(3): 543-547, 1983; Beck, A. & Teboulle, M. "A fast iterative shrinkage-thresholding algorithm for linear inverse problems." *SIAM J Imaging Sci* 2(1): 183-202, 2009.

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
