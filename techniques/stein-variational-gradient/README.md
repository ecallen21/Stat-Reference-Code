# Stein Variational Gradient Descent (Reference §47.87)

Liu & Wang (2016). Deterministic particle-based Bayesian inference.
Given target `log p(x)`, evolve n particles by

    φ(x_i) = (1/n) Σ_j [ k(x_j, x_i) ∇ log p(x_j) + ∇_{x_j} k(x_j, x_i) ]
    x_i ← x_i + ε · φ(x_i).

First term = weighted score push (particles chase modes); second =
repulsion (kernel gradient) that prevents collapse. Guaranteed to
decrease KL(q ‖ p) for small ε.

## Files

- `python/stein_variational_gradient.py` — from-scratch SVGD
  with RBF kernel + median-heuristic bandwidth. Demo (n=30
  particles started near origin, target = ½ N(−2, 1) + ½ N(+2, 1),
  500 iterations, ε=0.05):
  - initial mean ≈ 0.0, std ≈ 0.4
  - final mean ≈ −0.14, **std ≈ 2.16** (target √5 = 2.24)
  - particle split 53 % left mode / 47 % right (target 50/50).
- `r/stein_variational_gradient.R` — no first-class R port;
  `pyro`, `numpyro`, `torch` custom (Python).

## When to use

- **Bayesian posteriors** where a gradient is available (score
  functions from auto-diff).
- **Bayesian NNs / VAEs** — SVGD particles = alternative to MCMC.
- **Reinforcement learning** — Stein-Policy for exploration.
- **Amortized inference** — VAE + SVGD hybrids for posteriors.

## When NOT to use

- **No gradient available** — use random-walk MH or ABC.
- **Very high-dim x with few particles** — repulsion term
  underdetermined; add particle-augmentation tricks.
- **Discrete distributions** — SVGD needs continuous score;
  Stein-Point Markov chains handle discrete.

## Assumptions & caveats

- **Kernel bandwidth** — median heuristic on pairwise particle
  distances; degrades if particles collapse to a mode.
- **Step size ε** — line search or Adam-SVGD helps.
- **n particles** — as n → ∞, empirical dist → target;
  small-n approximations biased toward modes.
- **Non-convex log p** — SVGD can miss modes if all particles
  initialised near one; use annealed / diverse init.

## Related in this repo

- `variational-inference`, `hmc-nuts`, `hamiltonian-mc`,
  `mcmc-metropolis-hastings`, `stochastic-gradient-mcmc`,
  `laplace-approximation` — posterior-sampling toolkit.
- `bayesian-neural-network`, `mc-dropout`, `swag`,
  `deep-ensembles` — Bayesian deep-learning cousins.
- `score-matching`, `energy-based-models`,
  `diffusion-model` — score-function-based generative modelling.
- `particle-filter-smc` — sequential particle Bayes analogue.

## Run

```
python techniques/stein-variational-gradient/python/stein_variational_gradient.py
Rscript techniques/stein-variational-gradient/r/stein_variational_gradient.R
```

**Refs:** Liu, Q. & Wang, D. "Stein Variational Gradient Descent: A general purpose Bayesian inference algorithm." *NeurIPS*, 2016; Gorham, J. & Mackey, L. "Measuring sample quality with Stein's method." *NeurIPS*, 2015.

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
