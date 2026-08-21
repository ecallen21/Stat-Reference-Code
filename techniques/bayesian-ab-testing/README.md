# Bayesian A/B Testing (Reference §14.33)

Two Bernoulli arms `A` and `B` (conversion, click, sign-up) with unknown rates `p_A, p_B`. Put Beta priors, get Beta posteriors, report **full posteriors** rather than a single p-value.

## Setup

```
p ~ Beta(α₀, β₀)                        prior (Beta(1,1) = uniform is common)
y | n, p ~ Binomial(n, p)                observed successes
p | y ~ Beta(α₀ + y, β₀ + n − y)         conjugate posterior
```

## Quantities to report

- **P(B > A)** — direct probability that B has the higher rate. `> 0.95` is a common decision threshold.
- **Posterior lift** `p_B − p_A` and its 95% credible interval.
- **Posterior ratio** `p_B / p_A` for multiplicative interpretation.
- **Expected loss** under each choice (Chris Stucchio's formulation): `E[max(p_A − p_B, 0)]` if you pick B, and vice versa. Choose whichever has the smaller expected loss.

## Contrast with frequentist A/B

- **No peeking penalty** — posterior updates are valid at every step; no need for alpha-spending.
- **Interpretation** — "there is a 95% chance B is better" instead of "we reject a null we don't actually believe".
- **Handles unequal sample sizes** naturally.
- **Small sample-size** — informative priors stabilize the posterior when data is sparse.

## Files

- `python/bayesian_ab_testing.py` — Beta-Binomial posteriors + Monte Carlo comparisons (P(B > A), lift, ratio, expected loss). Demo (120/1000 vs 145/1000): P(B > A) = 0.95; posterior lift mean 0.025, 95% CI (−0.005, 0.055); expected loss if you pick B ≈ 0.0003 vs 0.025 for A.
- `r/bayesian_ab_testing.R` — same in base R via `rbeta`.

## When to use

- **Sequential / dashboard-style testing** where teams naturally peek at results.
- **Small-sample decisions** where the classical Fisher's exact / chi-square is too coarse.
- **Multiple conversion metrics** — separate Bayesian A/B for each; combine into a decision rule.
- **Value-of-information** analyses: expected loss ties directly to business cost of a wrong decision.

## Extensions

- **Multi-armed bandit** — Thompson sampling on the same posteriors.
- **Continuous outcomes** — Normal-Normal (see `conjugate-priors`).
- **Hierarchical A/B** — pooled across many similar experiments (`bayesian-hierarchical-models`).
- **Expected loss + minimum-effect stopping rules** — Ville-style bounded testing.

## Assumptions & caveats

- **Prior sensitivity** — Uniform Beta(1, 1) is often fine for `n ≥ 30`; for very small samples, an informative prior on plausible rates matters.
- **Independent observations** — no carryover effects, no bot traffic in A vs B.
- **SUTVA** — no spillover between arms (users in A shouldn't affect B).
- **Report the decision rule** explicitly: "stop when P(B > A) > 0.95 and expected loss < 0.001".

## Run

```
python techniques/bayesian-ab-testing/python/bayesian_ab_testing.py
Rscript techniques/bayesian-ab-testing/r/bayesian_ab_testing.R
```

**Refs:** Miller, E. "Formulas for Bayesian A/B testing." *Evan Miller blog*, 2015; Stucchio, C. *Bayesian A/B Testing at VWO*, 2015; Kruschke, J.K. *Doing Bayesian Data Analysis*, 2nd ed., Academic, 2015.

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
