# Hierarchical Reinforcement Learning: Options (Reference §28.x extra)

Sutton, Precup & Singh (1999). Introduce **temporally-extended actions**
called *options*, letting the top-level policy make fewer, more meaningful
decisions.

## An option

```
<I, π_o, β>
  I ⊆ S          initiation set — states where the option can start
  π_o(a | s)     intra-option policy (primitive-action policy while running)
  β(s) ∈ [0, 1]  termination probability at s
```

Top-level "policy over options" chooses an option; the option runs its
`π_o` until `β` triggers termination; the top level then re-selects.

## SMDP-Q-learning over options

The Markov chain of `(state, option)` decisions is a **Semi-Markov Decision
Process** (variable-duration transitions). Q-learning generalises:

```
Q(s, o) ← Q(s, o) + α · [Σ_k γ^k r_k + γ^K max_{o'} Q(s_K, o') − Q(s, o)]
```

where `K` is the number of primitive steps the option ran.

## Family

- **Options** (Sutton 1999) — the original.
- **MAXQ** (Dietterich 2000) — value-function decomposition over a task hierarchy.
- **Feudal Networks** (Vezhnevets 2017) — manager sets latent goals; worker acts.
- **Option-Critic** (Bacon 2017) — end-to-end learning of options via policy gradient.
- **HIRO** (Nachum 2018) — goal-conditioned hierarchy on continuous control.
- **DIAYN** (Eysenbach 2019) — unsupervised skill discovery via mutual-information objective.

## When to use

- **Long-horizon sparse-reward tasks** — options bridge the horizon.
- **Skill transfer** — options learned on one task reused on a related one.
- **Interpretable behaviour** — top-level policy names what's happening.
- **Hierarchical control in robotics** — high-level path planner + low-level motor policy.
- **NOT** when good options are hard to design or discover — misspecified options cripple the top-level agent.

## Files

- `python/hierarchical_rl_options.py` — SMDP-Q-learning over two hand-designed options ("walk left until 0" / "walk right until end") on an 8-state chain. Compared with primitive-action Q-learning:
  - **Primitive Q-learning** with ε = 0.2 **never finds the goal** in 15 episodes (50-step cap on each).
  - **HRL over options** hits +5.3 return **from episode 1** — one macro decision traverses the whole chain.
  - Final `Q(0, walk-right) = 5.3` vs `Q(0, walk-left) = 0` — clear preference.
- `r/hierarchical_rl_options.R` — `reticulate` + Python `ray[rllib]`, `garage / rlkit / cleanrl`; references to option-critic, feudal networks, HIRO, DIAYN.

## Assumptions & caveats

- **Option design is the hard part** — bad options make things worse than primitives.
- **Option-Critic** learns options + top-level policy jointly, avoiding hand-design.
- **Termination collapse** — learned options tend to terminate every step; regularise β.
- **Bootstrap discounting** — γ^K in the SMDP-Q backup; use the actual number of primitive steps.
- **Interruption / preemption** — allow the top level to preempt an option early.
- **Off-policy** intra-option learning is common (Precup 2000) — every primitive step gives an update.

## Related in this repo

- `reinforcement-learning-basics`, `dqn-deep-q-network`, `ppo-clipped` — flat-RL foundations.
- `exploration-strategies` — HRL is one way to solve hard exploration.
- `mdp-value-iteration` — planning in the flat SMDP.
- `hmm`, `markov-transition-models` — SMDPs generalise Markov chains with variable durations.

## Run

```
python techniques/hierarchical-rl-options/python/hierarchical_rl_options.py
Rscript techniques/hierarchical-rl-options/r/hierarchical_rl_options.R
```

**Refs:** Sutton, R.S., Precup, D. & Singh, S. "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning." *Artif. Intell.* 112, 181–211, 1999; Bacon, P.-L., Harb, J. & Precup, D. "The option-critic architecture." *AAAI*, 2017; Vezhnevets, A.S. et al. "FeUdal networks for hierarchical reinforcement learning." *ICML*, 2017.

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
