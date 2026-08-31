# Exploration Strategies (Reference §28.11)

The **exploration vs exploitation** trade-off is RL's oldest problem.
Naïve `ε-greedy` fails on hard-exploration tasks where the reward is
sparse or requires a long sequence of specific actions.

## Undirected exploration

- **ε-greedy** — random action w.p. `ε`; simple; O(2^n) episodes on chain-MDP.
- **Boltzmann / softmax** — sample `π ∝ exp(Q / τ)`; smoother; adapts to Q scale.
- **Noisy Nets** (Fortunato 2018) — parameter-space noise on the policy; usually beats ε-greedy on Atari.

## Optimism / uncertainty-based

- **UCB / UCB1** (Auer 2002) — `Q + c · √(log t / N_a)`; principled regret bounds in bandits.
- **UCB-V, KL-UCB, MBIE-EB** — variance-adjusted / distributional variants.
- **PSRL / Bootstrapped DQN** (Osband 2016) — posterior sampling from an ensemble.

## Intrinsic motivation

- **Count-based intrinsic reward** — `r_int(s) = β / √N(s)`; add to extrinsic reward.
- **Pseudo-counts** (Bellemare 2016) — density models generalise counts to continuous states.
- **ICM** (Pathak 2017) — intrinsic curiosity module; reward = prediction error of a learned forward model.
- **RND** (Burda 2018) — random network distillation; predict the output of a fixed random target network.
- **NGU / Agent57** (Badia 2020) — episodic + lifelong novelty; state-of-the-art on Atari hard-exploration games.
- **DIAYN / Empowerment** — mutual-information-based; learn diverse skills.

## Global / hard-exploration

- **Go-Explore** (Ecoffet 2019) — remember visited states, teleport back and continue; beats Atari Montezuma's Revenge without intrinsic rewards.
- **Novelty search / quality-diversity** — maintain an archive of diverse behaviours.

## When to use which

- **Dense reward, easy exploration**: ε-greedy or Boltzmann is fine.
- **Moderate hardness**: UCB / Noisy Nets.
- **Very sparse reward**: RND / ICM / NGU or Go-Explore.
- **Discovery / open-ended learning**: DIAYN / novelty search.
- **Multi-arm bandits**: Thompson sampling (see `multi-armed-bandits`).

## Files

- `python/exploration_strategies.py` — Chain-MDP benchmark (8 states, sparse +10 reward at the far end, 0 per step). Compare four exploration strategies over 500 episodes:
  - **ε-greedy (0.2)**: **never finds the goal** (first-success = None); mean return 0 for the last 50 episodes.
  - **Boltzmann (τ=0.5)**: first success at ep 0; mean return 10.
  - **UCB1 (c=2)**: first success at ep 1; mean return 10.
  - **Count-based intrinsic (β=0.5)**: first success at ep 5; mean return 10.
- `r/exploration_strategies.R` — no strong native R implementations; use `reticulate` + Python `stable-baselines3` (`EpsilonSchedule`, Noisy Nets), `cleanrl` reference implementations, and papers on RND / ICM / NGU / Agent57 / Go-Explore.

## Assumptions & caveats

- **Chain-MDP is a stress test**; real environments are usually easier.
- **Intrinsic rewards can dominate** the extrinsic signal — anneal `β` over training.
- **Non-stationary intrinsic reward** — as counts grow, the bonus shrinks; policy learning must be robust to this drift.
- **Curiosity in stochastic environments** (the "noisy TV" problem) — RND avoids this because the target is a deterministic random function; ICM does not.
- **Exploration coupled to representation** — good state features are prerequisite for good exploration; MAE / DINO / RND features help.
- **Combinations** — production Atari agents (R2D2, NGU, Agent57) stack multiple exploration signals + distributed data collection.

## Related in this repo

- `reinforcement-learning-basics`, `dqn-deep-q-network`, `ppo-clipped` — the RL algorithms these strategies plug into.
- `multi-armed-bandits` — the bandit / stateless special case.
- `bayesian-optimization` — a sibling optimisation-vs-exploration trade-off.

## Run

```
python techniques/exploration-strategies/python/exploration_strategies.py
Rscript techniques/exploration-strategies/r/exploration_strategies.R
```

**Refs:** Auer, P., Cesa-Bianchi, N. & Fischer, P. "Finite-time analysis of the multiarmed bandit problem." *Machine Learning* 47, 2002; Bellemare, M. et al. "Unifying count-based exploration and intrinsic motivation." *NeurIPS*, 2016; Pathak, D. et al. "Curiosity-driven exploration by self-supervised prediction (ICM)." *ICML*, 2017; Burda, Y. et al. "Exploration by random network distillation (RND)." *ICLR*, 2019; Badia, A.P. et al. "Never Give Up: learning directed exploration strategies (NGU)." *ICLR*, 2020.

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
