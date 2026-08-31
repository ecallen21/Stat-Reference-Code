# Soft Actor-Critic — SAC (Reference §28.x extra)

Haarnoja et al. (2018). Off-policy actor-critic that augments the reward
with a **policy-entropy bonus**, giving state-of-the-art sample efficiency
on continuous control (MuJoCo, D4RL) and robust exploration behaviour.

## Maximum-entropy RL

Standard RL maximises `𝔼[Σ r_t]`. SAC maximises:

```
J(π) = 𝔼_π [ Σ_t r_t + α · H(π(·|s_t)) ]
```

- `H(π)` is the entropy of the action distribution.
- `α` (temperature) trades exploration for exploitation.
- Optimal policy is a **soft-Boltzmann** over Q values: `π(a | s) ∝ exp(Q(s, a) / α)`.

## Soft Bellman backup

```
V_soft(s)  = α · log Σ_a exp(Q(s, a) / α)
Q(s, a)    ← r + γ · V_soft(s')
```

For continuous actions, replace the discrete `log Σ exp` with an expectation
sampled from `π`.

## SAC recipe

- **Two Q networks** (double-Q trick) → avoid overestimation: `Q_target = r + γ · [min_j Q_j(s', a') − α · log π(a' | s')]`.
- **Reparameterisation trick** for the Gaussian actor: `a = tanh(µ + σ · ε)`.
- **Learnable temperature** (Haarnoja 2018b): treat `α` as a Lagrange multiplier tuned to a target entropy `H̄`.
- **Replay buffer** — off-policy sample efficiency.

## When to use

- **Continuous control** — MuJoCo humanoid, robotic arm, drone, autonomous driving simulators.
- **Sample-efficient off-policy RL** — beats PPO by 5–10× on wall-clock samples on standard MuJoCo tasks.
- **Robust exploration** — entropy bonus prevents premature convergence.
- **NOT for very high-dimensional discrete action spaces** — SAC-Discrete works but PPO / DQN usually preferred.

## Files

- `python/sac_soft_actor_critic.py` — tabular **soft policy iteration** (the exact-model analogue of SAC). LineWorld demo with `α ∈ {0.1, 0.5, 2.0}` shows:
  - Small `α` → deterministic optimal policy `P(right) = 1.0`.
  - Large `α = 2.0` → softer policy `P(right)` from 0.71 to 0.95 — visible exploration bias.
  - `V_soft` grows with `α` because the entropy bonus adds to expected return.
- `r/sac_soft_actor_critic.R` — no native R support; `reticulate` + Python `stable-baselines3.SAC`, `cleanrl/sac_continuous_action.py`, `ray[rllib].SACConfig`, `d3rlpy.SAC`.

## Assumptions & caveats

- **Reward scale matters** — SAC's α interacts with reward magnitude; either normalise the reward or tune α (auto-tuned α makes this easier).
- **Target-network Polyak averaging** (τ = 0.005 typical) — no hard periodic sync as in DQN.
- **Discrete-action variant** — replaces the Gaussian actor with a categorical over actions; log-prob and entropy are simpler.
- **Off-policy** — samples from the replay buffer; benefits from big buffers (10⁶ typical) and multiple critic updates per policy update.
- **Reparameterisation** trick requires a re-parameterisable action distribution — Gaussian for continuous, categorical + Gumbel-softmax for discrete.
- **Robotics deployment** — SAC's entropy bonus tends to give smoother, less brittle policies than PPO.

## Related in this repo

- `actor-critic-a2c`, `ppo-clipped`, `dqn-deep-q-network` — RL neighbours.
- `mdp-value-iteration` — SAC's exact-model counterpart is soft policy iteration.
- `ddpg-td3` — deterministic-policy alternative for continuous control.
- `exploration-strategies` — entropy bonus is one exploration mechanism among many.

## Run

```
python techniques/sac-soft-actor-critic/python/sac_soft_actor_critic.py
Rscript techniques/sac-soft-actor-critic/r/sac_soft_actor_critic.R
```

**Refs:** Haarnoja, T. et al. "Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor." *ICML*, 2018; Haarnoja, T. et al. "Soft actor-critic algorithms and applications." *arXiv:1812.05905*, 2018; Christodoulou, P. "Soft actor-critic for discrete action settings." *arXiv:1910.07207*, 2019.

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
