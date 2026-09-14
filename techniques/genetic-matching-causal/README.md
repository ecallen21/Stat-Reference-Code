# Genetic Matching (Reference §47.327)

Diamond & Sekhon (2013). Search over a WEIGHT matrix `W` in
the generalised Mahalanobis distance:

```
d(x_i, x_j; W) = √((x_i − x_j)ᵀ W (x_i − x_j))
```

using a GENETIC ALGORITHM to MINIMISE the worst-case KS
statistic or SMD across covariates AFTER matching. Combines
propensity scores with covariates for better balance.

## Files

- `python/genetic_matching_causal.py` — Diagonal-weight
  variant of genetic matching. Compare on n=300 with unequal
  covariate shifts: baseline Mahalanobis vs 10-generation
  genetic search of 16 candidates. Genetic-search weights
  concentrate on covariates with the largest imbalance and
  reduce worst-case SMD by tuning W.
- `r/genetic_matching_causal.R` —
  `Matching::GenMatch`,
  `MatchIt::matchit(method='genetic')`, `rgenoud` (R);
  reticulate + Matching::GenMatch, from-scratch (Python).

## When to use

- **Observational studies** where propensity + Mahalanobis
  matching don't fully balance key covariates.
- **When investigator-defined weights** are hard to guess —
  genetic search discovers them.
- **Sekhon-style causal inference** pipelines in economics /
  political science.

## When NOT to use

- **Small samples** — GA overfits with few pairs.
- **Where full matching is preferred** — optimal-matching via
  min-cost flow beats greedy 1:1.
- **Very-high-dimensional X** — GA search space explodes;
  use regularised propensity + Mahalanobis instead.

## Assumptions & caveats

- **Fitness function** — max |SMD| or minimum KS p-value;
  choice affects the optimum.
- **Population size + generations** — Diamond-Sekhon
  recommend pop ≥ 1000, gens 500+ for real datasets.
- **Rgenoud** in R combines derivative-based local search
  with GA — usually beats vanilla GA.
- **Weight structure** — full covariance matrix `W`
  parameterisation is standard; diagonal (used here) is a
  simplification.

## Related in this repo

- `mahalanobis-distance-matching`,
  `propensity-score-matching`, `entropy-balancing`,
  `overlap-weighting`, `coarsened-exact-matching`,
  `inverse-probability-weighting`, `iptw` — companion
  balancing methods.

## Run

```
python techniques/genetic-matching-causal/python/genetic_matching_causal.py
Rscript techniques/genetic-matching-causal/r/genetic_matching_causal.R
```

**Refs:** Diamond, A. and Sekhon, J.S. "Genetic matching for estimating causal effects: a general multivariate matching method for achieving balance in observational studies." *Review of Economics and Statistics*, 95(3): 932-945, 2013.

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
