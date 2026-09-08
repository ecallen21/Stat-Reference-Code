# C51 - Distributional RL (Reference §47.124)

Bellemare, Dabney & Munos (2017). Instead of learning the SCALAR
Q(s, a) = E[Return], learn the DISTRIBUTION Z(s, a) over returns
via a categorical distribution on 51 fixed atoms `z₀ < … < z_{N−1}`:

    Z(s, a) = Σᵢ pᵢ(s, a) · δ_{zᵢ}
    Bellman target:  T Z(s, a) = R + γ · Z(s', a*)   (project onto atoms)
    Loss = cross-entropy between predicted p(s, a) and projection.

Rainbow (Hessel et al 2018) combines C51 + double Q + prioritised
replay + dueling + noisy nets + n-step.

## Files

- `python/distributional_rl_c51.py` — from-scratch categorical
  projection + empirical demonstration on a stochastic-reward
  Markov chain with returns 50/50 in {−1, +1}. As T grows the
  learned categorical converges to the true bimodal distribution
  (P(z<0) → 0.5). Also demonstrates atom projection preserving
  expectation: 0.5 δ(−1.3) + 0.5 δ(+0.8) → sum 1.000, E = −0.250.
- `r/distributional_rl_c51.R` — no first-class R port; `dopamine`,
  `stable-baselines3-contrib`, `torchrl` in Python.

## When to use

- **Atari-scale RL** where return distributions are skewed or
  bimodal — C51 was a strong DQN upgrade.
- **Risk-sensitive control** — CVaR / quantile policies from Z.
- **Reward-shaping diagnostics** — inspect return distributions
  per action.
- **Distributional off-policy evaluation**.

## When NOT to use

- **Simple tabular MDPs** — plain Q-learning suffices.
- **Continuous actions** — QR-DQN and IQN (quantile-regression
  distributional variants) scale better; DDPG / SAC for continuous
  control.
- **Real-time constraints** — categorical head adds cost.

## Assumptions & caveats

- **Atom range V_min / V_max** must cover possible returns; else
  bias.
- **N atoms** trades resolution vs compute; 51 is convention.
- **Cross-entropy target** projected onto atoms every update.
- **QR-DQN** (Dabney 2018) replaces categorical with quantile
  regression and avoids the range choice.

## Related in this repo

- `dqn-deep-q-network`, `mdp-value-iteration`,
  `actor-critic-a2c`, `ppo-clipped`, `sac-soft-actor-critic`,
  `ddpg-td3`, `prioritized-experience-replay`,
  `hierarchical-rl-options`, `exploration-strategies`,
  `gae-advantage-estimation` — RL toolkit.
- `bayesian-neural-network`, `deep-ensembles`, `swag`,
  `mc-dropout`, `evidential-deep-learning`,
  `conformal-classification`, `epistemic-aleatoric` —
  uncertainty-quantification cousins.
- `expectile-regression`, `bayesian-quantile-regression` —
  distributional regression neighbours.

## Run

```
python techniques/distributional-rl-c51/python/distributional_rl_c51.py
Rscript techniques/distributional-rl-c51/r/distributional_rl_c51.R
```

**Refs:** Bellemare, M.G., Dabney, W. & Munos, R. "A distributional perspective on reinforcement learning." *ICML*, 2017; Dabney, W., Ostrovski, G., Silver, D. & Munos, R. "Implicit quantile networks for distributional reinforcement learning." *ICML*, 2018.

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
