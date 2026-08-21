# Reinforcement Learning Basics (Reference §27.x extra)

Agent-in-environment learning: choose actions to maximise expected cumulative
reward. Two workhorse algorithms.

## Q-learning (Watkins 1989)

Off-policy tabular value learning:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · max_{a'} Q(s', a') − Q(s, a) ]
```

- **Off-policy** — learns about the greedy policy even while acting ε-greedy.
- **Guaranteed convergence** to Q* under mild conditions on `α, γ, ε`.
- **DQN** (Mnih 2013) generalises to a neural approximator + experience replay + target network.

## REINFORCE (Williams 1992)

On-policy policy-gradient:

```
θ ← θ + α · G_t · ∇_θ log π_θ(a_t | s_t)
```

- **On-policy** — must use the current policy's samples.
- **High variance** — subtract a baseline (value function) → **advantage actor-critic (A2C / A3C)**.
- **Trust-region variants** — TRPO, PPO stabilise the policy update.

## Family

| Family | Algorithms | When to use |
|---|---|---|
| **Value-based** | Q-learning, SARSA, DQN, Rainbow | discrete actions, sample efficiency, off-policy replay |
| **Policy gradient** | REINFORCE, A2C, A3C, PPO, TRPO | continuous actions, stochastic policies |
| **Actor-critic** | A2C, SAC, TD3 | continuous control, stable robotic policies |
| **Model-based** | Dyna-Q, MuZero, MBPO | sample-scarce environments |
| **Offline / batch** | BCQ, CQL, IQL | learning from logged data (no interaction) |
| **RLHF** | PPO on preference reward | LLM alignment (InstructGPT, Claude, Gemini) |

## When to use

- **Discrete decision problems with a simulator** — games, recommendation, dialogue policy.
- **Continuous control** — robotics, autonomous driving, industrial control.
- **Sequential clinical decision-making** — dynamic treatment regimes (with offline RL / TMLE for safety).
- **LLM fine-tuning from preferences** — RLHF or DPO.
- **NOT for one-shot supervised problems** — regression / classification alone is simpler and stronger.

## Files

- `python/reinforcement_learning_basics.py` — tabular Q-learning (ε-greedy) + REINFORCE (tabular softmax policy) from scratch on a 4-state LineWorld MDP (walk right to goal; reward −1 per step, +10 at goal). Both converge to "always right" policy in ~100–300 episodes with mean return 7.75 / 7.90 (near-optimal 8.0).
- `r/reinforcement_learning_basics.R` — `ReinforcementLearning`, `MDPtoolbox::mdp_Q_learning / mdp_policy_iteration`; `reticulate` + `gymnasium` + `stable-baselines3` (PPO, DQN, SAC, TD3, A2C), `ray[rllib]`.

## Assumptions & caveats

- **Exploration vs exploitation** — ε-greedy is a crude baseline; Boltzmann, UCB, curiosity-driven exploration (RND, ICM) are stronger.
- **Discount factor `γ`** matters — small `γ` shortsighted; `γ = 1` requires episodic termination.
- **Function approximation** breaks Q-learning's convergence guarantees; DQN adds target network + experience replay + reward clipping to stabilise.
- **Reward hacking** — the agent optimises what you *measure*, not what you *want*; RLHF for LLMs shows this vividly.
- **Sample complexity** is atrocious — millions of environment steps for even simple Atari games; use imitation learning, transfer learning, model-based, or offline RL when interaction is expensive.
- **Off-policy evaluation** is a research area of its own (IPS, weighted-IPS, doubly-robust); needed when you can't experiment on the deployed policy.

## Related in this repo

- `hmm`, `markov-transition-models` — the (fully observed) Markov-decision-process backbone.
- `adam-optimizer`, `deep-mlp-backprop`, `convolutional-nn` — the function-approximator neighbours.
- `contrastive-learning`, `transformer-decoder` — RLHF composes RL over LM policies.
- `tmle-doubly-robust`, `inverse-probability-weighting` — off-policy causal evaluation cousins.

## Run

```
python techniques/reinforcement-learning-basics/python/reinforcement_learning_basics.py
Rscript techniques/reinforcement-learning-basics/r/reinforcement_learning_basics.R
```

**Refs:** Watkins, C.J.C.H. "Learning from delayed rewards." PhD thesis, Cambridge, 1989; Williams, R.J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." *Machine Learning* 8, 229–256, 1992; Sutton, R.S. & Barto, A.G. *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018; Mnih, V. et al. "Human-level control through deep reinforcement learning." *Nature* 518, 529–533, 2015.

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
