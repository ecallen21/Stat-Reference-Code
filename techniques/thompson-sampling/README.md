# Thompson Sampling (Reference §47.60)

Thompson (1933). Bayesian multi-armed bandit strategy. Maintain a
posterior over each arm's mean reward; at each round:

    1. Draw θ_k ~ posterior_k for every arm k.
    2. Play argmax_k θ_k.
    3. Observe reward, update the posterior.

For Bernoulli arms with Beta(α, β) conjugate priors, draws are
trivial. Regret grows as O(log T) — matches the Lai-Robbins
lower bound (Agrawal-Goyal 2012).

## Files

- `python/thompson_sampling.py` — Beta-Bernoulli Thompson +
  UCB1 baseline. Demo (5 arms means [.30, .50, .55, .60, .70],
  T=5000):
  - Thompson plays best arm 4236/5000 times, regret 85.
  - UCB1     plays best arm 3797/5000 times, regret 180.
  Thompson has lower finite-time regret, matching theory.
- `r/thompson_sampling.R` — `contextual` (R);
  `scikit-multiflow`, `mabwiser`, from-scratch (Python).

## When to use

- **A/B/n testing with adaptive allocation** — send more traffic
  to winning arms.
- **News / ad recommendation** — bandit-slate variants.
- **Bayesian optimisation / hyperparameter tuning** — GP-Thompson.
- **Response-adaptive clinical trials** — ethical: allocate more
  patients to superior arms as evidence accumulates.

## When NOT to use

- **Fixed-horizon confirmatory RCT** — Type I error and inference
  become non-trivial (though group-sequential + Thompson exist).
- **Non-stationary rewards** — vanilla Thompson forgets; use
  discounting or sliding-window Thompson.
- **Adversarial rewards** — Exp3 / minimax is more robust.
- **Contexts without conjugacy** — use Bayesian bootstrap or
  particle Thompson.

## Assumptions & caveats

- **Prior specification** — Beta(1, 1) uniform is common; strong
  priors bias the exploration.
- **Independence across arms** — vanilla Thompson assumes it;
  contextual Thompson (with linear or GP models) handles side
  information.
- **Non-Bernoulli** — Beta-Bernoulli for {0, 1}; use Gaussian or
  gamma-Poisson conjugates for continuous / count rewards.
- **Regret is per-round** — cumulative regret grows unbounded but
  sublinearly.

## Related in this repo

- `multi-armed-bandits`, `exploration-strategies` — general MAB
  toolkit.
- `bayesian-ab-testing`, `bayesian-optimization` — Bayesian
  decision cousins.
- `mdp-value-iteration`, `dqn-deep-q-network`, `ppo-clipped` —
  full-RL extensions.
- `response-adaptive-randomization`, `platform-trial-design` —
  clinical-trial applications.

## Run

```
python techniques/thompson-sampling/python/thompson_sampling.py
Rscript techniques/thompson-sampling/r/thompson_sampling.R
```

**Refs:** Thompson, W.R. "On the likelihood that one unknown probability exceeds another in view of the evidence of two samples." *Biometrika* 25: 285-294, 1933; Agrawal, S. & Goyal, N. "Analysis of Thompson sampling for the multi-armed bandit problem." *COLT*, 2012.

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
