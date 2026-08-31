# QAP Network Regression (Reference §30.6)

**Krackhardt 1988** — permutation-based valid inference for regression
on dyadic (network) data.

## Why standard OLS fails on dyads

The `n(n-1)` dyad observations share nodes, so their errors are
correlated. Naive OLS SEs are anti-conservative.

## Quadratic Assignment Procedure

1. Vectorise `Y` (and each `X_k`) off-diagonal; fit OLS; record `β̂`.
2. For `B` permutations, permute the **node labels of `Y`** (row +
   column simultaneously); refit; get null distribution of `β̂`.
3. `p_k = ℙ_perm(|β_null_k| ≥ |β_obs_k|)`.

## Dekker-Krackhardt-Snijders "double semi-partialling"

For multi-covariate regression, permuting only `Y` breaks under
collinearity. Their DSP variant permutes the **residuals** of each
predictor's regression on the others; more robust in practice.

## When to use

- **Any regression where the response is a dyad-level variable**
  (frequency, tie strength, dissimilarity).
- **Ecological network** studies (species interactions).
- **Social networks** with covariates (kinship, geography, category
  homophily).

## When NOT to use

- **Node-level outcome regression** — QAP is designed for dyadic
  responses.
- **Very small n** — the permutation distribution is coarse.
- **Weighted continuous outcomes with heteroscedasticity** — the DSP
  variant still needs a valid variance model.

## Files

- `python/qap_network_regression.py` — from-scratch permutation-based
  QAP test on dyadic OLS. Demo: symmetric friendship matrix drives
  communication frequency; a random symmetric noise matrix is
  irrelevant. **friendship p = 0.000; noise p = 0.98** — exactly the
  expected pattern.
- `r/qap_network_regression.R` — `sna::netlm`, `statnet` (R);
  `netperm` (Python).

## Assumptions & caveats

- **Exchangeability under H0** — permuting node labels of `Y` is valid
  if `Y` and `X` are related only through the specified regression.
- **Number of permutations** — `B ≥ 500` for stable p-values;
  `B ≥ 5000` for publication.
- **Collinearity between predictors** — use Dekker DSP variant.
- **Directed / weighted networks** — QAP generalises; edge weights
  should be permuted with the labels.
- **Missing edges** — handle as `NA` in the vectorised regression, not
  as zeros.

## Related in this repo

- `bootstrap`, `permutation-test` (if present) — the resampling
  toolbox.
- `stochastic-block-model`, `ergm-exponential-random-graph`,
  `latent-space-network` — generative model alternatives (this batch).
- `network-descriptives`, `graph-descriptives` — the numeric-summary
  side of network analysis.

## Run

```
python techniques/qap-network-regression/python/qap_network_regression.py
Rscript techniques/qap-network-regression/r/qap_network_regression.R
```

**Refs:** Krackhardt, D. "Predicting with networks: nonparametric multiple regression analyses of dyadic data." *Social Networks*, 1988; Dekker, D., Krackhardt, D. & Snijders, T.A.B. "Sensitivity of MRQAP tests to collinearity and autocorrelation conditions." *Psychometrika*, 2007.

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
