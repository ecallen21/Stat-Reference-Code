# Generalised Advantage Estimation — GAE (Reference §28.12)

Schulman et al. (2016). Exponentially-weighted average of `n`-step
temporal-difference errors — the standard advantage estimator inside PPO,
A2C, TRPO, IMPALA.

## Formula

Given rewards `r_t` and value estimates `V(s_t)`:

```
δ_t     = r_t + γ V(s_{t+1}) − V(s_t)         (1-step TD residual)
A_t^GAE = Σ_{k=0}^{T−t−1} (γ · λ)^k · δ_{t+k}
returns  = A_t + V(s_t)                         (value-function training target)
```

## The `λ` knob

| `λ` | `A_t^GAE` | Bias | Variance |
|---|---|---|---|
| **0** | `δ_t` (pure 1-step TD) | high | low |
| **1** | `R_t − V(s_t)` (Monte-Carlo return) | zero | high |
| **~0.95** | interpolation | small | moderate |

Standard PPO / A2C hyperparameter: `λ = 0.95`, `γ = 0.99`.

## Why it matters

- **Reduces gradient variance** without introducing much bias.
- **Backward recursive computation** in `O(T)` — cheap.
- **Handles episode boundaries** via `nonterminal` mask on the recursion.
- **Numerical stability** — advantages are normalised across the batch before the policy update.

## When to use

- **On-policy actor-critic** (PPO, A2C, TRPO) — the standard companion.
- **Off-policy** — DQN / SAC / TD3 / DDPG don't use GAE; they use TD or Q-learning targets directly.
- **IMPALA / V-trace** — an off-policy correction to GAE with importance sampling; used in distributed RL.

## Files

- `python/gae_advantage_estimation.py` — from-scratch backward-recursion GAE. Demo (T=10 random rewards, decreasing value estimates, γ=0.99):
  - `λ = 0.0`: `Var(A) = 0.72` (biased, low variance).
  - `λ = 0.5`: `Var(A) = 1.10`.
  - `λ = 0.95`: `Var(A) = 2.97`.
  - `λ = 1.0`: `Var(A) = 3.50` (unbiased Monte-Carlo).
  - Sanity: `returns` at `λ=1` equal Monte-Carlo returns to 2e-16.
- `r/gae_advantage_estimation.R` — `reticulate` + Python `stable-baselines3` `RolloutBuffer.compute_returns_and_advantages`, `cleanrl/ppo.py`, `ray[rllib]`.

## Assumptions & caveats

- **Episode boundaries** — reset the recursion at each terminal; use a `dones` mask.
- **Bootstrap at the truncated horizon** — for rollouts that don't terminate, add `V(s_T)` as the bootstrap so the trailing sum is well-defined.
- **Advantage normalisation** — subtract mean, divide by std across the batch; standard PPO trick, reduces optimisation instability.
- **`γ · λ`** together control the effective horizon; increasing either lengthens it.
- **Bias increases** with model mis-specification — a badly-trained `V(s)` leaks bias into GAE at any `λ < 1`.
- **V-trace** (IMPALA) — the off-policy generalisation; adds importance-sampling ratios to correct for behaviour-vs-target policy mismatch.

## Related in this repo

- `ppo-clipped`, `actor-critic-a2c` — the algorithms GAE feeds.
- `reinforcement-learning-basics`, `dqn-deep-q-network` — foundational alternatives.
- `mdp-value-iteration` — the exact planning counterpart.

## Run

```
python techniques/gae-advantage-estimation/python/gae_advantage_estimation.py
Rscript techniques/gae-advantage-estimation/r/gae_advantage_estimation.R
```

**Refs:** Schulman, J. et al. "High-dimensional continuous control using generalized advantage estimation (GAE)." *ICLR*, 2016; Sutton, R.S. & Barto, A.G. *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018 (Chapter 12); Espeholt, L. et al. "IMPALA: scalable distributed deep-RL with importance weighted actor-learner architectures (V-trace)." *ICML*, 2018.

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
