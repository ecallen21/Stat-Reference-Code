# Noisy Networks for Exploration (Reference §47.146)

Fortunato et al. (2018). Replaces the ordinary linear layer
y = W x + b with a factorised-Gaussian noisy layer:

    W = μ_W + σ_W ⊙ ε_W,   b = μ_b + σ_b ⊙ ε_b,

where ε_W = f(ε_out) f(ε_in)ᵀ and ε_b = f(ε_out), with
f(x) = sgn(x) √|x|. Parameters (μ, σ) are learned by SGD; ε is
resampled every forward pass. Provides **state-dependent
exploration** that gradually amortises away as σ shrinks — a
strict improvement on decaying ε-greedy.

## Files

- `python/noisy_networks_exploration.py` — a from-scratch noisy
  linear layer + factorised-Gaussian sampling; trained on a
  contextual bandit (1-D context, 4 arms, arm 0 correct for
  ctx ≤ 0, arm 2 correct for ctx > 0).
  - First 500 steps mean reward: 0.73
  - Last 500 steps mean reward: **0.92**
  - Final σ_W mean magnitude: **0.019** (self-annealed).
  - Greedy accuracy after training: **95.3 %** (chance 25 %).
- `r/noisy_networks_exploration.R` — no native R port;
  `cleanrl` / `stable-baselines3` (Python) via `reticulate`.

## When to use

- **Deep RL** as a drop-in replacement for ε-greedy /
  Boltzmann exploration.
- **Rainbow DQN** — noisy nets is one of its six components.
- **Distributional RL** (C51, QR-DQN) as the exploration
  mechanism.

## When NOT to use

- **Continuous action** off-policy methods — parameter-noise variant
  (Plappert 2018) or entropy regularisation (SAC) is more standard.
- **Tabular / small state spaces** — plain ε-greedy is simpler and
  works fine.
- **On-policy algorithms** (A2C, PPO) — their stochastic policies
  already handle exploration.

## Assumptions & caveats

- **Initial σ_0** typically 0.5 / √fan_in; smaller = less
  exploration.
- **Factorised noise** (paper's default) uses O(in+out) random
  samples per step vs O(in·out) for full-Gaussian; matches full
  version's performance.
- **Gradient of L wrt σ** flows through both the sampled noise and
  the reparameterisation; PyTorch / JAX autograd handles this
  cleanly.
- **σ can go negative** — use its absolute value in the sample, or
  reparameterise as σ = softplus(ρ).

## Related in this repo

- `rainbow-dqn` — parent architecture using noisy nets as
  component #6.
- `exploration-strategies` — ε-greedy / UCB / Thompson sampling
  siblings.
- `bayesian-neural-network`, `mc-dropout` — Bayesian analogues of
  weight noise.
- `sac-soft-actor-critic` — entropy-regularised continuous-action
  alternative.

## Run

```
python techniques/noisy-networks-exploration/python/noisy_networks_exploration.py
Rscript techniques/noisy-networks-exploration/r/noisy_networks_exploration.R
```

**Refs:** Fortunato, M. et al. "Noisy networks for exploration." *ICLR*, 2018; Plappert, M. et al. "Parameter space noise for exploration." *ICLR*, 2018.

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
