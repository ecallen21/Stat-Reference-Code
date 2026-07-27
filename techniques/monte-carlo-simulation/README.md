# Monte Carlo Simulation (Reference §10.9)

Two canonical uses of Monte Carlo in applied statistics — both amount to *"simulate `N` datasets under a known process, run the analysis, aggregate."*

## 1. Power analysis

Under a specified **alternative** (chosen effect size, sample size, and analysis method), simulate many datasets, run the test, and compute the fraction of correct rejections.

```
power_hat  =  mean( p_value < α )        MC SE = √(p(1−p)/n_sim)
```

Compare to closed-form power formulas where they exist (e.g. `power.t.test`); MC gives the answer for any test/design where no closed form is available.

## 2. CI coverage

Under a known-parameter data-generating process, simulate many datasets, compute the CI method, and count how often the true parameter falls inside. Should equal `1 − α`. Coverage well below nominal → the CI method underestimates uncertainty for this DGP.

## Files

- `python/monte_carlo_simulation.py` — generic `power_simulation(sample_fn, test_fn, ...)` and `coverage_simulation(sample_fn, ci_fn, true_param, ...)` drivers with MC SE + Wilson CI on the empirical proportion.
- `r/monte_carlo_simulation.R` — same, plus cross-check against `stats::power.t.test` for the demo.

**Demo (Python):** empirical power of a Welch t-test at δ = 0.5, n = 30/group is 0.475 (MC SE 0.011); theoretical is 0.478. Empirical t-CI coverage at n = 10 is 0.945 with target 0.95.

## Assumptions

- Your `sample_fn` faithfully represents the DGP you want to study — MC is only as honest as the simulator.
- `n_sim` large enough that the MC SE is much smaller than the effect you're studying. For power near 0.5, MC SE at 2000 sims is ~0.01; for power near 0.05, it's ~0.005. Increase `n_sim` when the MC error would obscure the difference you care about.

## Run

```
python techniques/monte-carlo-simulation/python/monte_carlo_simulation.py
Rscript techniques/monte-carlo-simulation/r/monte_carlo_simulation.R
```

**Refs:** Metropolis, N. & Ulam, S. "The Monte Carlo method." *JASA* 44(247), 335–341, 1949; Ripley, B.D. *Stochastic Simulation*, Wiley, 1987; Efron, B. & Tibshirani, R.J. *An Introduction to the Bootstrap*, Chapman & Hall, 1993 (Ch. 6).

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
