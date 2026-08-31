# DDPG + TD3 (Reference §28.x extra)

Deterministic-policy actor-critic for continuous action spaces.

## DDPG (Lillicrap 2015)

Deep deterministic policy gradient:

```
actor:   µ_θ(s) : S → A         deterministic
critic:  Q_φ(s, a)
Q target: y = r + γ · Q_target(s', µ_target(s'))
critic loss: (Q(s, a) − y)²
actor loss:  − Q(s, µ(s))                                        (chain rule)
exploration: Gaussian or Ornstein-Uhlenbeck noise on the action
```

- Off-policy — uses a replay buffer.
- Extended DPG (Silver 2014) to deep-network policies.
- Notoriously **unstable**: overestimation bias + hyperparameter sensitivity.

## TD3 (Fujimoto 2018)

Twin Delayed DDPG fixes DDPG's overestimation via three tricks:

1. **Clipped Double Q** — two critics; use `min(Q₁, Q₂)` in the target.
2. **Delayed policy update** — update the actor once per `d` critic updates.
3. **Target-policy smoothing** — add clipped Gaussian noise to `µ_target(s')` in the target.

Result: much more stable than DDPG; matches or beats SAC on many MuJoCo tasks with fewer hyperparameters.

## When to use

- **Continuous control** — robotics, autonomous driving, drone, industrial control.
- **Deterministic policy required** — safety-critical, reproducible actions.
- **Off-policy sample efficiency** — beats PPO in wall-clock on standard MuJoCo.
- **Alternative to SAC** — SAC's entropy bonus gives smoother exploration; TD3 gives more deterministic behaviour at deploy.

## Files

- `python/ddpg_td3.py` — from-scratch DDPG + TD3 on a 1-D contextual bandit `r = −(a − s)²` (optimal action `a = s`, so linear actor θ* = 1). Uses a **quadratic-in-a** critic basis so the true Q can be represented. After 3000 episodes both converge to θ ≈ 0.75 (near optimal 1.0); mean reward −0.12 (optimal ~ −noise² = 0.09 with exploration).
- `r/ddpg_td3.R` — `reticulate` + `stable-baselines3.DDPG / TD3`, `cleanrl/td3_continuous_action.py`, `ray[rllib].DDPGConfig`.

## Assumptions & caveats

- **Overestimation** is DDPG's main failure mode; TD3's min-Q fixes most of it.
- **Actor updates without min-Q** — TD3 updates the actor using `Q_1` only (not the min), which is intentional; using min in the actor loss over-regularises.
- **Exploration noise** must be scaled to the action range; OU noise is more autocorrelated than Gaussian and often preferred in robotics.
- **Reward normalisation** — DDPG is more sensitive to reward magnitude than SAC because it lacks the entropy bonus scaling.
- **Target-network Polyak τ** — τ = 0.005 typical (very slow updates); large τ destabilises training.
- **Convergence of the demo** — quadratic critic + linear actor is a small illustrative case; on real MuJoCo, TD3 needs 100k–1M steps for good policies.

## Related in this repo

- `sac-soft-actor-critic` — stochastic-policy off-policy alternative.
- `actor-critic-a2c`, `ppo-clipped` — on-policy alternatives.
- `dqn-deep-q-network` — discrete-action off-policy counterpart.
- `exploration-strategies` — OU / Gaussian noise families.

## Run

```
python techniques/ddpg-td3/python/ddpg_td3.py
Rscript techniques/ddpg-td3/r/ddpg_td3.R
```

**Refs:** Silver, D. et al. "Deterministic policy gradient algorithms (DPG)." *ICML*, 2014; Lillicrap, T.P. et al. "Continuous control with deep reinforcement learning (DDPG)." *ICLR*, 2016; Fujimoto, S., van Hoof, H. & Meger, D. "Addressing function approximation error in actor-critic methods (TD3)." *ICML*, 2018.

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
