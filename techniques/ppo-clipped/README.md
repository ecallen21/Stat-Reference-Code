# Proximal Policy Optimization — PPO (Reference §28.5)

Schulman et al. (2017). The most-used deep-RL algorithm — the workhorse for
Atari, robotics, and RLHF for LLMs.

## Clipped surrogate objective

Let `r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)` be the importance ratio
between the current and behaviour policies. PPO maximises:

```
L^CLIP(θ) = 𝔼_t [ min( r_t · A_t,  clip(r_t, 1 − ε, 1 + ε) · A_t ) ]
```

- **Clipping** on `r_t` prevents any single gradient step from moving the policy too far from `π_old`, achieving a trust-region-like effect without the expensive conjugate-gradient / KL-constraint machinery of TRPO.
- `ε = 0.1–0.3` typical; `ε = 0.2` is the classical default.
- Full loss: `L = L^CLIP − c_V · MSE(V, R) + c_H · H(π)`.

## Recipe

1. **Rollout**: collect `T` steps with the current policy `π_{θ_old}`.
2. **Compute** discounted returns `R_t` and advantages `A_t` (usually via GAE — see `gae-advantage-estimation`).
3. **Normalise** advantages within the batch.
4. **`K` optimisation epochs** of mini-batch gradient ascent on `L`.
5. Update `θ_old ← θ`, repeat.

## When to use

- **Continuous control** — MuJoCo, robotics, self-driving simulators.
- **Discrete control** — Atari (with A2C-style vector envs), board games.
- **RLHF for LLMs** — InstructGPT, ChatGPT, Claude, LLaMA-Chat, DeepSeek all used PPO-style updates on a preference-model reward.
- **When A2C / DDPG are unstable** — PPO's clipping tolerates larger LRs and longer rollouts.

## Files

- `python/ppo_clipped.py` — from-scratch tabular PPO with softmax actor + tabular critic. Full rollout → Monte-Carlo returns → normalised advantages → K=4 epochs of clipped-surrogate update. On LineWorld (5 states) after 100 iterations:
  - `P(right)` > 0.96 in all non-terminal states.
  - `V = [4.54, 6.16, 7.96, 10.0, 0.0]`.
  - Mean return over last 20 iterations = 6.90 (near-optimal 7.0).
- `r/ppo_clipped.R` — `torch::nn_module`; Python `stable-baselines3.PPO`, `cleanrl/ppo.py`, `ray[rllib].PPOTrainer`, TRL `PPOTrainer / DPOTrainer / GRPOTrainer` for LLM fine-tuning.

## Assumptions & caveats

- **On-policy** — must collect rollouts under the current policy each iteration; less sample-efficient than DQN.
- **Advantages** — the version without GAE (as in the demo) has higher variance; production PPO always uses GAE(`λ`).
- **Clip fraction** — monitor the fraction of samples where the clip is active; ~10–20% is healthy, > 30% often means the LR is too large.
- **KL divergence** — track `KL(π_old || π)` per epoch; some PPO variants early-stop when KL exceeds a threshold.
- **Value clipping** — clip the value-function update analogously to stabilise the critic.
- **LR + entropy schedules** — decay `lr` and `c_H` over training; cosine or linear schedules standard.
- **Multi-GPU / distributed** — `ray[rllib].PPOTrainer` scales to thousands of vector envs.

## Related in this repo

- `reinforcement-learning-basics`, `actor-critic-a2c`, `dqn-deep-q-network` — foundations and alternatives.
- `gae-advantage-estimation` — the standard companion for computing `A_t`.
- `rlhf-preferences` — how PPO composes with a reward model on human-preference data.
- `adam-optimizer`, `deep-mlp-backprop` — training-loop pairings.

## Run

```
python techniques/ppo-clipped/python/ppo_clipped.py
Rscript techniques/ppo-clipped/r/ppo_clipped.R
```

**Refs:** Schulman, J. et al. "Proximal policy optimization algorithms." *arXiv:1707.06347*, 2017; Schulman, J. et al. "Trust region policy optimization (TRPO)." *ICML*, 2015; Shao, Z. et al. "DeepSeekMath: pushing the limits of mathematical reasoning in open language models (GRPO)." *arXiv:2402.03300*, 2024.

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
