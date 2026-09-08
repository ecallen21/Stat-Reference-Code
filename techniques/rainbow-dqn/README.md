# Rainbow DQN (Reference §47.145)

Hessel et al. (2018). Combines six independent extensions of DQN
into a single agent that outperforms each alone on Atari:

1. **Double DQN** (van Hasselt 2016) — decouples action selection
   / evaluation.
2. **Prioritized replay** (Schaul 2016) — samples transitions with
   probability ∝ |TD error|^α.
3. **Dueling networks** (Wang 2016) — separate value / advantage
   heads.
4. **Multi-step returns** (Sutton 1988) — n-step TD targets.
5. **Distributional RL** (Bellemare 2017) — predicts return
   distribution, not just mean.
6. **Noisy networks** (Fortunato 2018) — learnable weight noise
   replaces ε-greedy.

## Files

- `python/rainbow_dqn.py` — tabular illustration of the
  gradient-free three components on a 7-state chain MDP (goal at
  right end, ±reward). Uses Double-Q + prioritised replay + n-step
  returns:
  - Learned greedy policy after 400 episodes: `[1, 1, 1, 1, 1, 1, 0]`
    (right-right-…-right, i.e. **optimal**).
  - n-step = 1 avg return (last 50 eps): 0.942
  - n-step = 3: 0.939
  - n-step = 5: 0.934
  - (Deep dueling / distributional / noisy variants require neural
    nets; see the noisy-networks-exploration sibling.)
- `r/rainbow_dqn.R` — no native R implementation; recommends
  `stable-baselines3` / `cleanrl` (Python) via `reticulate`.

## When to use

- **Model-free deep RL** on discrete-action environments (Atari,
  toy control, grid-world).
- Whenever you'd otherwise use DQN — Rainbow is a near-strict
  improvement.

## When NOT to use

- **Continuous action spaces** — use SAC / TD3 / PPO.
- **Very small state spaces** — plain tabular Q-learning is
  simpler and provably optimal.
- **On-policy** requirements — Rainbow is off-policy (uses replay).

## Assumptions & caveats

- **Hyperparameter interactions** — each component adds knobs
  (n, alpha, beta for PER; sigma for noisy; support / atoms for
  distributional). The paper's ablation shows PER + multi-step
  are the biggest wins.
- **Distributional support** [V_min, V_max] must contain the true
  return range.
- **Noisy-net sigma** anneals implicitly through gradient descent.
- **Replay-buffer size** dominates memory usage.

## Related in this repo

- `dqn-deep-q-network` — the baseline Rainbow improves on.
- `prioritized-experience-replay` — component #2 in isolation.
- `noisy-networks-exploration` — component #6 in isolation.
- `sac-soft-actor-critic`, `ddpg-td3`, `ppo-clipped` — sibling
  deep-RL algorithms.

## Run

```
python techniques/rainbow-dqn/python/rainbow_dqn.py
Rscript techniques/rainbow-dqn/r/rainbow_dqn.R
```

**Refs:** Hessel, M. et al. "Rainbow: Combining improvements in deep reinforcement learning." *AAAI*, 2018; van Hasselt, H., Guez, A. & Silver, D. "Deep reinforcement learning with double Q-learning." *AAAI*, 2016; Schaul, T. et al. "Prioritized experience replay." *ICLR*, 2016; Bellemare, M. G., Dabney, W. & Munos, R. "A distributional perspective on reinforcement learning." *ICML*, 2017.

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
