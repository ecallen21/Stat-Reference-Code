# Model-Based Reinforcement Learning (Reference §28.7)

Learn (or use) a **model** of the environment's dynamics `P(s' | s, a)` and
reward `R(s, a, s')`, then **plan** with that model to reduce the number of
real environment interactions needed.

## Dyna-Q (Sutton 1990)

The classical template. For each real step `(s, a, r, s')`:

1. **Real update** — Q-learning on the observed transition.
2. **Record** `(r, s')` in the model.
3. **Planning** — repeat `k` times: sample a random previously-seen `(s, a)`, imagine `(r̂, ŝ')` from the model, apply a Q-learning update.

Each real transition thus contributes `1 + k` Q updates, dramatically
reducing sample complexity in tabular / small-state MDPs.

## Deep MBRL family

| Method | Key idea |
|---|---|
| **World Models** (Ha-Schmidhuber 2018) | VAE + RNN world model + evolved linear controller |
| **PETS** (Chua 2018) | Probabilistic ensemble of dynamics + trajectory sampling |
| **MBPO** (Janner 2019) | Short model rollouts to augment a SAC replay buffer |
| **Dreamer v1/v2/v3** (Hafner 2020–23) | Latent RSSM world model + policy trained inside imagined trajectories |
| **MuZero** (Schrittwieser 2020) | MCTS in a learned latent world model — no simulator needed |
| **TD-MPC / TD-MPC2** (Hansen 2022) | Latent model + short-horizon MPC + Q-learning |

## When to use

- **Sample-scarce environments** — robotics, protein folding, plasma control.
- **Known dynamics** — MPC / trajectory optimisation for autonomous driving, drones.
- **Planning under uncertainty** — probabilistic models + trajectory sampling.
- **Not** when the model is very hard to learn accurately; model bias then makes MBRL worse than model-free.

## Files

- `python/model_based_rl.py` — from-scratch tabular Dyna-Q with 10 planning steps per real step, compared side-by-side with plain Q-learning on LineWorld (5 states). Both reach the optimal return of 7.0; Dyna-Q gets there in noticeably fewer real episodes.
- `r/model_based_rl.R` — `ReinforcementLearning` + custom model; Python deep MBRL references (World Models, PETS, MBPO, Dreamer, MuZero, TD-MPC).

## Assumptions & caveats

- **Model bias hurts** — planning with a bad model produces bad policies. Ensembles + short imagination horizons (MBPO / PETS) mitigate.
- **Compounding errors** — long rollouts diverge from the true trajectory; keep imagined rollouts short.
- **Deterministic vs stochastic models** — deterministic overfits; probabilistic ensembles capture epistemic uncertainty.
- **Latent world models** (Dreamer, MuZero, TD-MPC) train the model in an unsupervised way alongside the policy; state-of-the-art on DMC / Atari.
- **Model-based vs model-free** trade-off is one of sample efficiency vs asymptotic performance — MBRL wins early, model-free eventually catches up.
- **Planning depth `k`** in Dyna-Q — trades wall-clock per real step for sample efficiency.

## Related in this repo

- `reinforcement-learning-basics`, `mdp-value-iteration`, `dqn-deep-q-network`, `actor-critic-a2c`, `ppo-clipped` — model-free counterparts.
- `monte-carlo-tree-search` — planning with a known / learned model.
- `hmm`, `markov-transition-models` — transition-model families.
- `bayesian-optimization` — a related "learn a surrogate, plan with it" recipe for optimisation.

## Run

```
python techniques/model-based-rl/python/model_based_rl.py
Rscript techniques/model-based-rl/r/model_based_rl.R
```

**Refs:** Sutton, R.S. "Integrated architectures for learning, planning, and reacting based on approximating dynamic programming." *ICML*, 1990; Ha, D. & Schmidhuber, J. "World models." *NeurIPS*, 2018; Janner, M. et al. "When to trust your model: model-based policy optimization." *NeurIPS*, 2019; Hafner, D. et al. "Mastering diverse domains through world models (Dreamer v3)." *Nature Communications*, 2023.

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
