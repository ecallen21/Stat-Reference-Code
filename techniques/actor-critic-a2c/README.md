# Advantage Actor-Critic — A2C (Reference §28.4)

Policy-gradient RL with a **critic** that learns a value function used as
a variance-reduction baseline.

## The update

```
actor:   π_θ(a | s)        softmax over discrete actions or Gaussian for continuous
critic:  V_φ(s)

advantage:      A_t = r_t + γ V(s_{t+1}) − V(s_t)              (1-step TD)
policy loss:    L_π = − 𝔼_t [ log π(a_t | s_t) · A_t ]         (A_t treated as constant)
value loss:     L_V = 𝔼_t [ (r_t + γ V(s_{t+1}) − V(s_t))² ]
entropy bonus:  L_H = − 𝔼_t H(π(·|s_t))                        (encourages exploration)

L = L_π + c_v · L_V + c_H · L_H
```

## Why the critic helps

- **Baseline variance reduction** — subtracting `V(s)` from the return leaves the expected gradient unchanged (Sutton et al. 2000) but greatly reduces variance.
- **Bootstrap** — 1-step TD (`r + γ V`) reduces the delay before the policy sees learning signal, compared to Monte-Carlo REINFORCE which needs a full episode.
- **Bias-variance tradeoff** — n-step / GAE(`λ`) trades between the two.

## Family

- **A2C** — synchronous multi-worker actor-critic; Mnih 2016.
- **A3C** — asynchronous variant with lock-free updates.
- **PPO** (see `ppo-clipped`) — trust-region / clipped-surrogate policy update.
- **SAC** — continuous control with soft entropy objective.
- **IMPALA** — distributed importance-weighted actor-critic (Espeholt 2018).

## When to use

- **Discrete or continuous action spaces** with reasonable dimensionality.
- **On-policy** — samples must come from the current policy; not sample-efficient.
- **Standard baseline** to try before PPO on control tasks.
- **Not for sparse-reward hard-exploration** without intrinsic motivation (see `exploration-strategies`).

## Files

- `python/actor_critic_a2c.py` — from-scratch tabular A2C with softmax policy + TD(0) value + entropy bonus on LineWorld (5 states). After 300 episodes:
  - Learned `P(right) > 0.99` in all non-terminal states.
  - Learned `V(0)=4.5, V(1)=6.1, V(2)=7.8, V(3)=9.8`.
  - Mean return over last 20 episodes = 6.90 (optimal 7.0).
- `r/actor_critic_a2c.R` — `torch::nn_module` (manual); Python `stable-baselines3.A2C`, `cleanrl/a2c.py`, `ray[rllib].A2CTrainer`.

## Assumptions & caveats

- **Learning-rate ratio** — actor and critic often need different LRs (critic learns faster).
- **Entropy coefficient** — too large blurs the policy; too small collapses to a bad deterministic policy.
- **Normalising advantages** across a batch is standard in production PPO / A2C — reduces gradient scale variance.
- **1-step TD is biased** — GAE(`λ`) (see `gae-advantage-estimation`) smoothly trades bias for variance.
- **Discrete vs continuous** — softmax head for discrete; Gaussian or Beta head for continuous with a re-parameterisation trick.
- **Sample efficiency** — on-policy A2C uses each transition once; DQN + replay uses it many times.

## Related in this repo

- `reinforcement-learning-basics`, `mdp-value-iteration`, `dqn-deep-q-network` — foundations.
- `ppo-clipped`, `gae-advantage-estimation` — modern policy-gradient improvements.
- `deep-mlp-backprop`, `adam-optimizer` — the training-loop pairings.
- `exploration-strategies` — intrinsic-motivation add-ons for hard-exploration tasks.

## Run

```
python techniques/actor-critic-a2c/python/actor_critic_a2c.py
Rscript techniques/actor-critic-a2c/r/actor_critic_a2c.R
```

**Refs:** Sutton, R.S. et al. "Policy gradient methods for reinforcement learning with function approximation." *NeurIPS*, 2000; Mnih, V. et al. "Asynchronous methods for deep reinforcement learning (A3C)." *ICML*, 2016; Espeholt, L. et al. "IMPALA: scalable distributed deep-RL with importance weighted actor-learner architectures." *ICML*, 2018.

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
