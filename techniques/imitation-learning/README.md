# Imitation Learning (Reference §28.8)

Learn a policy from **expert demonstrations** rather than from reward
signals. Useful when the reward is hard to specify or dangerous to explore.

## Behavioural cloning (BC)

Straight supervised learning on `D = {(s_i, a_i^expert)}`:

```
min_θ  𝔼_(s, a) ~ D [ − log π_θ(a | s) ]
```

- Trivial to train — classification / regression.
- **Distribution shift** is the big problem: the learner visits states outside the expert's distribution and small errors compound (Ross-Bagnell 2010).

## DAgger (Ross-Gordon-Bagnell 2011)

Iteratively query the expert on the learner's own state distribution:

```
for iter in 1..N:
  roll out π_θ, collect visited states S_new
  query expert for a_new = expert(S_new)
  D ← D ∪ {(s, a_new) : s ∈ S_new}
  retrain π_θ on D
```

Provably matches expert performance under mild assumptions.

## Inverse RL / GAIL

- **MaxEnt IRL** (Ziebart 2008) — recover a reward function that explains expert behaviour, then RL against it.
- **GAIL** (Ho-Ermon 2016) — adversarial imitation: discriminator distinguishes expert from learner; policy trained by PPO to fool it. Scales to high-dim MuJoCo / Atari.
- **AIRL** (Fu 2018) — GAIL variant that recovers a shaped reward.

## When to use

- **Autonomous driving** — cheaper / safer than reward-shaping (Waymo, Wayve, Comma AI).
- **Robot manipulation** — kinesthetic teaching → BC → optional RL fine-tune.
- **LLM SFT** — supervised fine-tuning on human demonstrations is exactly BC over tokens (InstructGPT, LLaMA-Chat).
- **Warm start for RL** — BC to get a decent starting policy, then RL fine-tune (offline RL, RLHF).
- **Not** when demonstrations are sparse or the expert is only slightly better than random.

## Files

- `python/imitation_learning.py` — from-scratch tabular BC + DAgger on a slippery LineWorld (drift = 15%; expert policy = "always right"). Demo:
  - **BC** (5 rollouts, 30 data points): mean return 5.91, `P(right) ≈ 0.89` per state.
  - **DAgger** (8 iterations, 79 data points): mean return 5.89, `P(right) ≈ 0.95` — sharper policy at the states the learner actually visits.
  Both learn the correct action; DAgger's advantage grows on harder environments where distribution shift matters more.
- `r/imitation_learning.R` — `reticulate` + Python `imitation` (BC / DAgger / GAIL / AIRL), `stable-baselines3-contrib`, `d3rlpy`.

## Assumptions & caveats

- **DAgger requires an interactive expert** — you need to query them on new states. Often expensive (human) or impossible (only historical logs).
- **BC compounding error** — the learner's early mistakes take it to states the expert never showed, where it has no signal.
- **Covariate shift** matters — a state distribution that shifts even a little at training time makes BC brittle.
- **Reward-free imitation** doesn't recover the underlying preferences of the expert; IRL / RLHF do.
- **Multi-modal experts** — different experts may prefer different actions at the same state; a mode-averaging BC gives poor policies. Use mixture-density-network policies or expert-clustered BC.
- **Modern practice** — BC is the base; RL fine-tune (PPO / SAC) on top only when demonstrations are exhausted.

## Related in this repo

- `reinforcement-learning-basics`, `dqn-deep-q-network`, `actor-critic-a2c`, `ppo-clipped` — the RL neighbours.
- `offline-rl` — a related "learn from a fixed dataset" setting.
- `rlhf-preferences` — imitation-style pipeline for LLM alignment.
- `transfer-learning` — related "warm start" recipe on labelled data.

## Run

```
python techniques/imitation-learning/python/imitation_learning.py
Rscript techniques/imitation-learning/r/imitation_learning.R
```

**Refs:** Pomerleau, D.A. "ALVINN: an autonomous land vehicle in a neural network." *NeurIPS*, 1989; Ross, S., Gordon, G.J. & Bagnell, D. "A reduction of imitation learning and structured prediction to no-regret online learning (DAgger)." *AISTATS*, 2011; Ho, J. & Ermon, S. "Generative adversarial imitation learning (GAIL)." *NeurIPS*, 2016.

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
