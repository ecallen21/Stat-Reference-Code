# Multi-Armed Bandits (Reference §28.1)

**Sequential decision-making** with `k` arms, each yielding an unknown-mean
reward. Balance **exploration** (learning about arms) vs **exploitation**
(pulling the current best).

## Regret

Standard performance measure over `T` rounds:

```
R_T = T · µ* − Σ_t r_t
```

Optimal algorithms achieve `R_T = O(log T)` (Lai-Robbins lower bound).

## Three classical algorithms

| Algorithm | Rule | Regret |
|---|---|---|
| **ε-greedy** | argmax `μ̂` w.p. `1 − ε`; uniform w.p. `ε` | linear in `T` (bad); `ε_t = 1/t` recovers `O(log T)` |
| **UCB1** (Auer 2002) | argmax `μ̂_a + √(2 log t / n_a)` | `O(log T)` |
| **Thompson sampling** (Thompson 1933) | sample from posterior on each `μ_a`; pick argmax | `O(log T)`; often best empirically |

## Contextual bandits

Add a context `x_t ∈ ℝᵈ` per step; expected reward per arm is `θ_aᵀ x_t`.
Standard algorithm: **LinUCB** (Li 2010) — linear regression per arm + UCB
on predicted mean. Neural / deep contextual bandits use MLPs and Thompson
sampling via dropout or ensembles.

## When to use

- **A/B / multi-arm testing** with real-time reward feedback (news headlines, recommendations).
- **Clinical adaptive trials** — allocate more patients to the more-promising arm.
- **Online advertising / recommendation** — contextual bandit + LinUCB.
- **Sequential hyperparameter search** — bandit-flavoured Successive Halving / Hyperband.
- **Not for pure exploration** — use best-arm identification / pure-exploration algorithms.

## Files

- `python/multi_armed_bandits.py` — from-scratch ε-greedy, UCB1, Thompson sampling on a Bernoulli bandit. Demo (5 arms, `μ = [0.20, 0.35, 0.55, 0.40, 0.30]`, T=2000):
  - **ε-greedy (0.1)**: cum reward 995, cum regret 115.
  - **UCB1**: cum reward 992, cum regret 138.
  - **Thompson (Beta)**: cum reward 1076, cum regret **18.7** — an order of magnitude better.
  All three identify arm 2 as the best.
- `r/multi_armed_bandits.R` — `contextual::Simulator + Agent + Policy(EpsilonGreedy, UCB1, Thompson, LinUCB)`; Python `mabwiser`, `contextualbandits`, `vowpalwabbit --cb`.

## Assumptions & caveats

- **Stationary rewards** in the classical formulation; non-stationary bandits need discounting or windowing (D-UCB, SW-UCB).
- **Delayed feedback** — real advertising / RL settings; use delayed-reward bandits or importance-weighted updates.
- **Cold start** — Thompson needs a prior; a Beta(1, 1) uniform prior is safe.
- **Regret bound assumptions** — Bernoulli / bounded rewards; Gaussian bandits use UCB-V; heavy-tailed rewards need robust variants.
- **Best-arm identification** vs regret minimisation are different objectives; racing / successive-elimination algorithms are the pure-exploration counterparts.
- **Contextual bandits** need identifiable rewards; if the same context always gives the same arm-reward, the problem collapses to supervised regression.

## Related in this repo

- `reinforcement-learning-basics` — the general RL framework this specialises.
- `bayesian-optimization` — a related Thompson-sampling-flavoured approach for continuous action spaces.
- `bayesian-ab-testing` — the two-arm Bayesian analogue.
- `laplace-approximation`, `variational-inference` — approximate posteriors that make Thompson feasible on complex likelihoods.

## Run

```
python techniques/multi-armed-bandits/python/multi_armed_bandits.py
Rscript techniques/multi-armed-bandits/r/multi_armed_bandits.R
```

**Refs:** Robbins, H. "Some aspects of the sequential design of experiments." *Bull. AMS* 58(5), 527–535, 1952; Auer, P., Cesa-Bianchi, N. & Fischer, P. "Finite-time analysis of the multiarmed bandit problem." *Machine Learning* 47, 235–256, 2002; Thompson, W.R. "On the likelihood that one unknown probability exceeds another in view of the evidence of two samples." *Biometrika* 25, 285–294, 1933; Lattimore, T. & Szepesvári, C. *Bandit Algorithms*, Cambridge UP, 2020.

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
