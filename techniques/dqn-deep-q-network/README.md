# Deep Q-Network — DQN (Reference §28.3)

Q-learning with a **neural-network approximator** and two stabilisers that
made deep RL practical for the first time (Mnih 2013, 2015):

- **Experience replay** — store `(s, a, r, s', done)` in a buffer, sample minibatches to break temporal correlation.
- **Target network** — a copy of Q with frozen weights (synced every `N` steps) used to form the TD target `y = r + γ · max_{a'} Q_target(s', a')`.

## Loss

```
L = (y − Q_θ(s, a))²
```

- Sample a mini-batch from the replay buffer, compute `y` with the target network, take an SGD step on the online network.
- Sync target ← online every `N` gradient steps (or use a Polyak averaged copy).

## Rainbow (Hessel et al. 2018)

The Atari-era SOTA — combined six independently-published improvements:

| Improvement | Idea |
|---|---|
| **Double DQN** | separate networks for argmax and Q-value evaluation |
| **Duelling DQN** | `Q = V + A − mean(A)` two-stream decomposition |
| **Prioritised replay** | sample high-TD-error transitions more often |
| **Multi-step returns** | `n`-step Bellman target reduces bias |
| **Categorical (C51)** | model the whole return distribution, not just the mean |
| **Noisy Nets** | parameter-space noise for exploration |

## When to use

- **Discrete action spaces** — Atari, board games, discrete control.
- **Off-policy learning from historical data** — replay is a natural fit for offline RL warm-starts.
- **Sample efficiency vs on-policy** — DQN reuses each transition many times.
- **Not for continuous actions** — use DDPG / TD3 / SAC (see `reinforcement-learning-basics`).
- **Not for very high-dimensional continuous states** — PPO usually more stable.

## Files

- `python/dqn_deep_q_network.py` — from-scratch DQN with numpy MLP Q, replay buffer, and target network. Demo on LineWorld (5 states, 2 actions, right = +10 at goal, −1 per step): learns the "always right" policy; Q values grow monotonically toward the goal (state 3 →  Q = 9.99 for right); mean return over last 20 episodes = 6.35 (near-optimal 7.0 for the 4-step path from state 0).
- `r/dqn_deep_q_network.R` — `torch::nn_module` (manual); Python `stable-baselines3.DQN`, `cleanrl/dqn.py`, `ray[rllib]`.

## Assumptions & caveats

- **The deadly triad** (Sutton-Barto) — off-policy + function approximation + bootstrapping can diverge. Target networks + replay + gradient clipping are the mitigations.
- **Overestimation bias** — max is biased upward under noise; use Double DQN.
- **Exploration** — `ε`-greedy is the default; annealed from 1.0 to 0.05 over training. Noisy Nets and intrinsic-motivation methods (see `exploration-strategies`) do better on hard-exploration tasks.
- **Replay buffer size** — too small over-fits recent trajectories; typical Atari uses 10⁶.
- **Reward clipping** — for Atari, clip to `[−1, +1]`; keeps the Q scale bounded across games.
- **Frame skipping / stacking** — standard Atari preprocessing (4-frame skip, 4-frame stack).

## Related in this repo

- `reinforcement-learning-basics`, `mdp-value-iteration` — foundations.
- `actor-critic-a2c`, `ppo-clipped` — on-policy alternatives.
- `deep-mlp-backprop`, `adam-optimizer`, `dropout-batchnorm` — the training-loop pairings.
- `offline-rl` — recipes for replay-only training when interaction is impossible.

## Run

```
python techniques/dqn-deep-q-network/python/dqn_deep_q_network.py
Rscript techniques/dqn-deep-q-network/r/dqn_deep_q_network.R
```

**Refs:** Mnih, V. et al. "Playing Atari with deep reinforcement learning." *arXiv:1312.5602*, 2013; Mnih, V. et al. "Human-level control through deep reinforcement learning." *Nature* 518, 529–533, 2015; Hessel, M. et al. "Rainbow: combining improvements in deep reinforcement learning." *AAAI*, 2018.

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
