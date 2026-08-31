# Prioritised Experience Replay — PER (Reference §28.x extra)

Schaul et al. (2016). Sample transitions from a replay buffer with
probability **proportional to their TD-error magnitude**, so the network
sees informative transitions more often.

## Sampling probabilities

```
P(i) ∝ p_i^α ,       p_i = |TD-error_i| + ε
```

- `α = 0` → uniform replay.
- `α = 1` → full prioritisation (informative transitions dominate).
- `α = 0.6` typical.

## Importance-sampling correction

Because sampling is no longer uniform, the loss gradient is biased. Correct
with importance-sampling (IS) weights:

```
w_i = ( (1 / N) / P(i) )^β
```

- Annealed `β: 0.4 → 1.0` over training.
- Multiply per-sample gradients / losses by `w_i`.
- Normalise `w_i / max(w)` for numerical stability.

## Data structure

The standard efficient implementation is a **sum-tree** (Fenwick tree):
`O(log N)` update and sampling. The demo uses a simple linear array
because it's clearer to read.

## When to use

- **DQN / Rainbow / distributional RL** — 2–4× sample efficiency on Atari.
- **Any off-policy RL with a replay buffer** — SAC, TD3, offline RL warm-starts.
- **Hard-exploration tasks** — rare informative transitions get more weight.
- **NOT with on-policy algorithms** (PPO, A2C) — they don't use a replay buffer.

## Files

- `python/prioritized_experience_replay.py` — from-scratch `PrioritisedBuffer` with `α`, `β`, IS weights, and update. Toy dataset with 90% low-TD (~0.05) and 10% high-TD (~5.0) transitions. Demo shows:
  - Mean |TD-error| **under PER** (α = 0.6): 3.7.
  - Mean |TD-error| **under uniform**: 0.45.
  - **~8.3× more informative sampling**.
  - IS weights range 0.015–1.0 (with β = 1.0), correctly down-weighting over-sampled points.
- `r/prioritized_experience_replay.R` — `reticulate` + Python `stable-baselines3-contrib`, `ray[rllib]`, `cleanrl/dqn_atari_per.py`.

## Assumptions & caveats

- **Bias / variance trade-off** — full prioritisation with no IS weights collapses to the highest-TD subset; IS weights restore an unbiased gradient at `β = 1`.
- **Sum-tree is the correct data structure** at scale; linear-scan sampling is `O(N)` and unusable past 10⁵ transitions.
- **Priority staleness** — old priorities may not reflect current TD-error; update priorities every time you use a sample.
- **Priorities near zero** get `ε` (~1e-3) added to avoid never-sampling.
- **Interacts with target-network updates** — priorities computed against the online net differ from those against the target net; both work in practice.
- **Rank-based variant** (also from Schaul 2016) uses the rank of |TD| rather than its magnitude; more robust to outliers.

## Related in this repo

- `dqn-deep-q-network` — the primary user of PER.
- `sac-soft-actor-critic`, `ddpg-td3` — off-policy actor-critics that also use replay.
- `offline-rl` — the replay buffer is the whole dataset.
- `imitation-learning` — related "learn from replayed transitions" family.

## Run

```
python techniques/prioritized-experience-replay/python/prioritized_experience_replay.py
Rscript techniques/prioritized-experience-replay/r/prioritized_experience_replay.R
```

**Refs:** Schaul, T. et al. "Prioritized experience replay." *ICLR*, 2016; Hessel, M. et al. "Rainbow: combining improvements in deep reinforcement learning." *AAAI*, 2018; Kapturowski, S. et al. "Recurrent experience replay in distributed reinforcement learning (R2D2)." *ICLR*, 2019.

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
