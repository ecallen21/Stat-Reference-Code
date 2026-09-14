# Mahalanobis Distance Matching (Reference §47.326)

Rubin (1980); Rosenbaum & Rubin (1985). Match treated to
control units by the Mahalanobis distance:

```
d(x_i, x_j) = √((x_i − x_j)ᵀ S⁻¹ (x_i − x_j))
```

where `S` is the pooled covariance of covariates. Often
combined with a CALIPER (max acceptable distance) and / or with
propensity-score matching.

## Files

- `python/mahalanobis_distance_matching.py` — 1:1 NN
  Mahalanobis matching with caliper 0.5. n=600 with treatment-
  induced covariate shifts (0.5, 0.3, −0.2). Standardised
  mean differences drop from ~0.5 to under 0.05 after matching
  — well below Cochrane's 0.1 balance threshold.
- `r/mahalanobis_distance_matching.R` —
  `MatchIt::matchit(distance='mahalanobis')`, `Matching`,
  `optmatch` (R); `scipy.spatial.distance.cdist`,
  `causalmatch`, from-scratch (Python).

## When to use

- **Small number of continuous covariates** — Mahalanobis
  works well when d < ~10.
- **After PS matching** as a fine-tuning step for balance on
  key covariates.
- **When covariates are correlated** — Mahalanobis outperforms
  Euclidean.

## When NOT to use

- **Very high-dimensional covariates** — Mahalanobis
  distances get noisy; use propensity scores.
- **Categorical / mixed-type covariates** — need Gower-style
  distances or coarsened exact matching.
- **Extreme imbalance** — matching may drop many treated
  units.

## Assumptions & caveats

- **Cov(X) invertible** — regularise with ridge when d ≈ n.
- **Caliper choice** — Rubin recommends 0.25 of pooled SD;
  0.5 in the demo is loose.
- **1:1 vs 1:k matching** — 1:k improves variance at the
  cost of possible bias.
- **Replacement** — with-replacement matching preserves
  more treated units but complicates variance.

## Related in this repo

- `propensity-score-matching`, `overlap-weighting`,
  `entropy-balancing`, `coarsened-exact-matching`,
  `inverse-probability-weighting`, `iptw` — related causal
  balancing.
- `genetic-matching-causal` — the tuned-weight extension.

## Run

```
python techniques/mahalanobis-distance-matching/python/mahalanobis_distance_matching.py
Rscript techniques/mahalanobis-distance-matching/r/mahalanobis_distance_matching.R
```

**Refs:** Rubin, D.B. "Bias reduction using Mahalanobis-metric matching." *Biometrics*, 36(2): 293-298, 1980; Rosenbaum, P.R. and Rubin, D.B. "Constructing a control group using multivariate matched sampling methods that incorporate the propensity score." *Am. Stat.*, 39(1): 33-38, 1985.

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
