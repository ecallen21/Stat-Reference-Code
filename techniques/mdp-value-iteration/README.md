# MDP Value Iteration and Policy Iteration (Reference §28.2)

Solve finite Markov decision processes exactly by dynamic programming.

## Bellman optimality

For an MDP `(S, A, P, R, γ)`:

```
V*(s)  = max_a  Σ_{s'} P(s'|s, a) · [R(s, a, s') + γ V*(s')]
π*(s)  = argmax_a  Σ_{s'} P(s'|s, a) · [R(s, a, s') + γ V*(s')]
```

## Value iteration

```
V_{k+1}(s) = max_a Σ_{s'} P(s'|s, a) · [R + γ V_k(s')]
```

- Contraction with rate `γ` — converges geometrically.
- Extract greedy policy from the final `V`.

## Policy iteration

Alternate:

1. **Policy evaluation** — solve `V^π = R_π + γ P_π V^π` (linear system).
2. **Policy improvement** — greedy policy w.r.t. the current `V^π`.

Converges in `≤` finitely many steps for finite MDPs (fewer iterations than VI, but each iteration is more expensive because of the linear solve).

## Linear programming

An LP formulation with `|S|` variables minimises `Σ_s V(s)` subject to
Bellman consistency constraints. Convergence proof is easy; solvers scale to
a few thousand states.

## When to use

- **Small, fully-known MDPs** — teaching, verification, ground-truth for RL algorithms.
- **Gridworlds / inventory control / operations research**.
- **Dynamic programming approximation** for planning under uncertainty when transition probabilities are estimated from data.
- **NOT for large / continuous state spaces** — use function-approximation RL (DQN, PPO, SAC).

## Files

- `python/mdp_value_iteration.py` — from-scratch VI + PI on a 4×4 gridworld with a goal cell at (3, 3). Both algorithms converge in 7 iterations to the same optimal policy: walk down / right toward the goal. `V` at cells adjacent to the goal is exactly 10.0 (immediate goal reward).
- `r/mdp_value_iteration.R` — `MDPtoolbox::mdp_value_iteration / mdp_policy_iteration / mdp_LP`, `ReinforcementLearning::ReinforcementLearning`.

## Assumptions & caveats

- **Full model needed** — VI and PI assume `P` and `R` are known. RL algorithms (Q-learning, DQN, PPO) learn from samples.
- **State-space size** — `|S|` grows exponentially in problem features; use function approximation.
- **Discount `γ`** — small `γ` gives shortsighted policies; `γ = 1` requires guaranteed episode termination.
- **Precision** — VI tolerance `ε` should account for the effective horizon `1 / (1 − γ)`.
- **Stochastic vs deterministic policies** — VI / PI return deterministic; for stochastic MDPs a soft version (soft-Bellman / MaxEnt IRL) is used.
- **Partially observed** — POMDPs need belief-state MDPs; exact solutions rarely feasible past small dimensions.

## Related in this repo

- `reinforcement-learning-basics` — the sample-based counterpart.
- `dqn-deep-q-network` — the function-approximation extension.
- `hmm`, `markov-transition-models` — the transition-model neighbours.
- `bayesian-hierarchical-models` — an alternative uncertainty-aware planning approach.

## Run

```
python techniques/mdp-value-iteration/python/mdp_value_iteration.py
Rscript techniques/mdp-value-iteration/r/mdp_value_iteration.R
```

**Refs:** Bellman, R. *Dynamic Programming*, Princeton UP, 1957; Howard, R.A. *Dynamic Programming and Markov Processes*, MIT Press, 1960; Puterman, M.L. *Markov Decision Processes: Discrete Stochastic Dynamic Programming*, Wiley, 1994.

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
