# Hoeffding's D Independence Test (Reference §47.133)

Hoeffding (1948). Rank-based independence test that detects ANY
monotonic OR non-monotonic dependence (unlike Pearson / Spearman
which need monotonicity). Statistic based on the bivariate
empirical distribution vs the product of marginals.

## Files

- `python/hoeffding_d_independence.py` — from-scratch D
  statistic (Hollander-Wolfe form) + permutation p-value. Demo
  (n=120):
  - independent:  D = +0.00,  p = 0.74,  Spearman ρ = −0.06
  - linear:       D = +0.33,  p < 0.005, Spearman ρ = +0.85
  - quadratic (U): D = +0.19,  p < 0.005, **Spearman ρ = −0.15**
  - sinusoid:     D = +0.10,  p < 0.005, Spearman ρ = −0.38.
  Hoeffding D detects the quadratic dependence Spearman misses.
- `r/hoeffding_d_independence.R` — `Hmisc::hoeffd`,
  `energy::dcor.test` (R); `hyppo`, from-scratch (Python).

## When to use

- **General independence testing** without assuming monotonicity.
- **Diagnosing "why" correlation is zero** — nonlinear relations.
- **Screening in causal discovery / feature selection**.
- **Companion to Pearson / Spearman / Kendall** — triangulate.

## When NOT to use

- **Very large n** with high-dim data — kernel HSIC / distance
  correlation scale better.
- **Only monotonic dependence** — Spearman is simpler and
  interpretable.
- **Tiny n** — permutation-based p-value needed; asymptotic
  distribution slow to converge.

## Assumptions & caveats

- **Ranks** — invariant to monotone transformations of both
  variables.
- **Ties**: modified estimator handles them; classical form assumes
  continuous distributions.
- **D ≥ 0** under H_A (dependence); under H_0 it hovers near zero.
- **Alternatives**: distance correlation (Székely 2007), HSIC
  (Gretton 2005), MIC (Reshef 2011).

## Related in this repo

- `pearson-correlation`, `spearman-rank-correlation`,
  `kendalls-tau`, `distance-correlation` — correlation cousins.
- `hsic-independence`, `mmd-two-sample-test`,
  `mutual-information`, `conditional-mutual-info`,
  `f-divergences`, `kl-divergence` — dependence / divergence
  measures.
- `permutation-tests`, `friedmans-h-statistic` — nonparametric
  test tools.
- `causal-discovery-pc`, `notears-dag-learning` — independence-
  based causal discovery.

## Run

```
python techniques/hoeffding-d-independence/python/hoeffding_d_independence.py
Rscript techniques/hoeffding-d-independence/r/hoeffding_d_independence.R
```

**Refs:** Hoeffding, W. "A non-parametric test of independence." *Ann Math Stat* 19(4): 546-557, 1948; Hollander, M. & Wolfe, D.A. *Nonparametric Statistical Methods*, 2nd ed., Wiley, 1999.

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
