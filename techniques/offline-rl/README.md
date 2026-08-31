# Offline Reinforcement Learning (Reference §28.9)

Learn a policy from a **fixed dataset** `D = {(s, a, r, s')}` — no
environment interaction allowed. The key difficulty: naive off-policy
methods (Q-learning, DQN) extrapolate beyond `D`'s state-action support and
produce dangerously overconfident value estimates.

## Why naive off-policy fails

The Q-learning target `y = r + γ · max_{a'} Q(s', a')` maxes over **all**
actions, including actions the behaviour policy never took at `s'`.
For those out-of-distribution actions, `Q` is unconstrained by data and
gets pushed up by bootstrapping — a chain of overestimations that grows
with iterations.

## Conservative Q-Learning (Kumar 2020)

Add a penalty that keeps `Q(s, a)` low at OOD actions:

```
L_CQL = ½ (Q − target)² + α · [ log Σ_a exp Q(s, a)  −  𝔼_{a ~ D} Q(s, a) ]
```

- **First penalty term** — pushes down the log-sum-exp of Q over all actions.
- **Second** — anchors Q at dataset actions.
- Net effect: OOD Q values sit **below** in-distribution Q values; greedy policy stays close to the behaviour policy.

## Family

| Method | Idea |
|---|---|
| **BC** | supervised on `(s, a)` — best upper bound of the behaviour policy |
| **BCQ** (Fujimoto 2019) | conditional VAE proposes in-distribution actions + Q ensemble |
| **CQL** (Kumar 2020) | pessimism penalty on OOD Q |
| **IQL** (Kostrikov 2021) | expectile regression + advantage-weighted BC — SOTA on D4RL |
| **TD3+BC** (Fujimoto 2021) | TD3 + simple BC regulariser; SOTA-competitive |
| **AWAC** (Nair 2020) | advantage-weighted actor critic |

## When to use

- **Historical / logged data only** — dynamic treatment regimes, adaptive experiments, dialogue policies from logs.
- **Safety-critical domains** — no exploration allowed (autonomous driving, medical trials).
- **Warm start for online RL** — initialise from an offline policy, then interact.
- **NOT** when the dataset has poor coverage of the useful action space; garbage in → garbage out.

## Files

- `python/offline_rl.py` — from-scratch tabular Q-learning on a fixed dataset with and without a CQL-style penalty. Demo (LineWorld, right-biased behaviour policy giving 95% right; dataset has essentially no "left" data):
  - **Naive Q-learning**: Q(s, L) = 0 for all `s` (never updated); policy chooses R (correct for this dataset, but Q is not calibrated on OOD).
  - **CQL**: Q(s, L) < 0 for OOD "left" actions — the penalty successfully suppresses OOD values; Q(s, R) unchanged at its Bellman-optimal value.
- `r/offline_rl.R` — `reticulate` + `d3rlpy` (CQL, BCQ, IQL, TD3+BC, AWAC), `ray[rllib]` offline API.

## Assumptions & caveats

- **Dataset coverage caps performance** — offline RL cannot invent actions the dataset never contained.
- **Distributional shift** — evaluating the learned policy in the environment can behave very differently from the training-time offline metric.
- **Hyperparameter tuning is hard** — no online rollouts to score candidates; use OPE (off-policy evaluation) or held-out log-likelihood.
- **CQL's `α`** trades performance vs safety; too large collapses to BC.
- **Pessimism vs optimism** — CQL / BCQ / TD3+BC are pessimistic; IQL learns a value that's just above the behaviour policy — more optimistic, often better.
- **Reward specification** must match the dataset; historical logs often have implicit rewards (e.g. click-through) that need care.

## Related in this repo

- `reinforcement-learning-basics`, `dqn-deep-q-network`, `actor-critic-a2c`, `ppo-clipped` — online counterparts.
- `imitation-learning` — the reward-free special case.
- `inverse-probability-weighting`, `tmle-doubly-robust` — off-policy evaluation neighbours from causal inference.
- `landmark-analysis`, `joint-longitudinal-survival` — sequential-clinical-decision-making context.

## Run

```
python techniques/offline-rl/python/offline_rl.py
Rscript techniques/offline-rl/r/offline_rl.R
```

**Refs:** Levine, S. et al. "Offline reinforcement learning: tutorial, review, and perspectives on open problems." *arXiv:2005.01643*, 2020; Fujimoto, S., Meger, D. & Precup, D. "Off-policy deep reinforcement learning without exploration (BCQ)." *ICML*, 2019; Kumar, A. et al. "Conservative Q-Learning for offline reinforcement learning (CQL)." *NeurIPS*, 2020; Kostrikov, I., Nair, A. & Levine, S. "Offline reinforcement learning with implicit Q-learning (IQL)." *ICLR*, 2022.

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
