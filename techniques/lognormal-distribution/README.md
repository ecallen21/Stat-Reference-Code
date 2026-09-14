# Log-Normal Distribution (Reference §47.390)

`X = exp(Y)` with `Y ~ N(μ, σ²)`. PDF:

```
f(x) = 1 / (x σ √(2π)) · exp(−(log x − μ)² / (2 σ²))
E[X] = exp(μ + σ²/2)
Var  = (exp(σ²) − 1) · exp(2μ + σ²)
```

Right-skewed, heavier tail than Gaussian. Ubiquitous in income
distributions, particle size, biological growth, environmental
concentration, tissue-dose distributions.

## Files

- `python/lognormal_distribution.py` — MLE via log-transform
  + Gaussian mean/std. Recovers `(μ, σ)` from n=5 000 to 3
  decimals and matches empirical mean to E[X] closed form
  across three parameter regimes.
- `r/lognormal_distribution.R` — `stats::dlnorm` /
  `rlnorm` / `plnorm`, `MASS::fitdistr`,
  `fitdistrplus::fitdist` (R); `scipy.stats.lognorm`,
  from-scratch (Python).

## Where else it appears in this repo

- `parametric-survival`, `accelerated-failure-time` —
  log-normal AFT.
- `deep-survival-network` — DL survival with log-normal head.
- `bayesian-linear-regression` — log-transformed
  right-skewed outcomes.
- `gamma-regression`, `tweedie-glm-regression`, `gamlss` —
  GLM alternatives for positive skewed data.
- `extreme-value-theory` — log-normal has intermediate tail
  (not in GEV MDA).

## Assumptions & caveats

- **Positivity** — `X > 0` strict.
- **Log-transform trick** — MLE reduces to Gaussian MLE on
  `log X`; watch for zero-inflated data (add small ε or use
  hurdle model).
- **Bias of exp(μ̂)** — E[X] estimator with plug-in is biased;
  Finney's bias correction available.
- **Confusion with Gaussian** — log-normal often mistaken
  for Gaussian at small σ; do a QQ or Shapiro test on `log X`.

## Run

```
python techniques/lognormal-distribution/python/lognormal_distribution.py
Rscript techniques/lognormal-distribution/r/lognormal_distribution.R
```

**Refs:** Aitchison, J. and Brown, J.A.C. *The Lognormal Distribution*, Cambridge Univ Press, 1957; Limpert, E., Stahel, W.A. and Abbt, M. "Log-normal distributions across the sciences." *BioScience*, 51: 341-352, 2001.

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
