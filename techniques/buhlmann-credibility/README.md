# Bühlmann Credibility (Reference §24.20)

Bühlmann (1967). Classical actuarial framework predicting a group's
future losses from a weighted average of the group's own experience
and the collective mean:

    P_g = Z_g · X̄_g + (1 − Z_g) · μ
    Z_g = n_g / (n_g + K)
    K   = σ²_within / τ²_between

Higher `n_g` → more credibility to own experience. Lower within
variance vs between variance → more credibility to group.

Equivalent to a random-intercept BLUP; exactly the linear-Bayes
estimator under Normal-Normal.

## Files

- `python/buhlmann_credibility.py` — MoM estimation of `σ²`, `τ²`,
  `K`, then per-group credibility factor `Z_g` and shrinkage
  prediction from scratch. Demo (G=60, small groups, large within
  vs small between): K̂ = 7.84; Z ranges 0.28-0.64; Bühlmann MSE
  18.5 vs raw group mean MSE 28.5 → 35 % reduction.
- `r/buhlmann_credibility.R` — `actuar::cm`, `ChainLadder`,
  `lme4::lmer` BLUP (R); `chainladder`, `statsmodels.MixedLM`,
  from-scratch (Python).

## When to use

- **Insurance rate-making** — per-policyholder / per-territory
  loss forecasting.
- **Small-cell prediction with a natural population mean** — any
  hierarchical setting.
- **Actuarial reserving** — combined with Bornhuetter-Ferguson /
  chain-ladder methods.
- **Sports analytics / classroom / hospital ratings** — same math.

## When NOT to use

- **Zero between-group variance** — Z = 0; all shrunk to mean;
  Bühlmann adds nothing over the pooled estimator.
- **Very large n per group** — Z ≈ 1; group mean is fine.
- **Non-Gaussian / heavy-tailed losses** — extend to
  Bühlmann-Straub or non-normal Bayesian.

## Assumptions & caveats

- **Within / between decomposition** relies on second moments; MoM
  can give negative `τ²` estimates → clip to 0.
- **Homogeneity across groups** — Bühlmann assumes common `σ²`;
  use Bühlmann-Straub for heteroskedasticity.
- **BLUP equivalence** — `Z_g` is exactly the shrinkage weight in
  a linear mixed model with random intercepts.

## Related in this repo

- `bayesian-hierarchical-models`, `linear-mixed-models`,
  `james-stein-shrinkage` — statistical cousins.
- `fay-herriot-small-area`, `poisson-gamma-empirical-bayes`,
  `empirical-bayes` — small-area shrinkage siblings.

## Run

```
python techniques/buhlmann-credibility/python/buhlmann_credibility.py
Rscript techniques/buhlmann-credibility/r/buhlmann_credibility.R
```

**Refs:** Bühlmann, H. "Experience rating and credibility." *ASTIN Bulletin*, 4(3): 199-207, 1967; Bühlmann, H. & Gisler, A. *A Course in Credibility Theory and its Applications*, Springer, 2005.

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
