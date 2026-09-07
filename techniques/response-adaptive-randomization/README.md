# Response-Adaptive Randomization (RAR) (Reference §44.13)

Wei & Durham (1978); Berry et al. (2010). Modifies future
randomisation probabilities based on accumulated outcomes so more
participants receive the arm that is performing better.

## Common flavours

| Scheme | Update rule |
|---|---|
| Play-the-winner (Wei-Durham urn) | Non-parametric: success adds ball for the same arm |
| RPW | Randomised play-the-winner |
| Thompson / BAR | `P(next = A) = Pr(θ_A > θ_B | data)` under posterior |
| Fixed 1:1 | Static randomisation (baseline) |

## Trade-off

Fewer participants assigned to inferior arm (ethical appeal) **vs.**
lower power / larger sample size to reach a fixed type-1 / type-2
error.

## Files

- `python/response_adaptive_randomization.py` — from-scratch
  Wei-Durham urn + Thompson-BAR + fixed baseline over 300
  simulated two-arm trials. Demo (true rates 0.30, 0.45; N=400):
  fixed 50% / 50% / 89% power; Wei-Durham 44 / 56% / 89% power;
  Thompson BAR 13 / 87% / 68% power — Thompson strongly shifts to
  the better arm at a small power cost.
- `r/response_adaptive_randomization.R` — `adaptr`, `BAR`, `pipe`
  (R); from-scratch (Python).

## When to use

- **Rare-disease / small-n trials** — patient ethics motivate
  minimizing exposure to inferior arm.
- **Adaptive Phase II** — pick winners early with BAR.
- **Multi-arm dose ranging** — accelerate allocation to promising
  doses.

## When NOT to use

- **Regulatory Phase III with strict FDA type-I control** — hybrid
  designs and pre-specified stopping rules preferred.
- **Time trends / drift** — RAR reallocates dynamically; a trend in
  disease severity biases arm comparisons unless block-adjusted.
- **Very small samples** — early estimates are noisy; RAR can lock
  in the wrong arm (Villar-Bowden-Wason 2015 caution).

## Assumptions & caveats

- **No time trend** — implicit; add stratification by calendar
  block or use covariate-adjusted response-adaptive designs.
- **Accurate priors** — Thompson depends on posteriors; poor priors
  cause under- or over-allocation.
- **Reporting** — FDA guidance (2019 adaptive-design) requires
  full pre-specification and simulation-based operating
  characteristics.
- **Power comparison** — always simulate the specific design and
  effect size; nominal alpha/beta don't transfer directly.

## Related in this repo

- `bayesian-ab-testing`, `multi-armed-bandits` — same underlying
  Thompson / UCB toolkit.
- `sequential-analysis`, `always-valid-inference` — sequential
  monitoring cousins.
- `mde-sample-size`, `cuped-variance-reduction` — fixed-design
  variance-reduction alternatives.
- `platform-trial-design` — extends RAR to multi-arm platform.

## Run

```
python techniques/response-adaptive-randomization/python/response_adaptive_randomization.py
Rscript techniques/response-adaptive-randomization/r/response_adaptive_randomization.R
```

**Refs:** Wei, L.J. & Durham, S. "The randomized play-the-winner rule in medical trials." *JASA*, 73(364): 840-843, 1978; Berry, D.A. et al. *Bayesian Adaptive Methods for Clinical Trials*, CRC, 2010.

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
