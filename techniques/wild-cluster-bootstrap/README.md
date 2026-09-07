# Wild Cluster Bootstrap (Reference §12.15)

Cameron, Gelbach & Miller (2008); MacKinnon & Webb (2018).
Bootstrap-based inference for clustered errors that outperforms the
asymptotic cluster-robust sandwich (CR1) when the number of clusters
`G` is small (< 40) or clusters are unbalanced.

## Scheme (for OLS `y = Xβ + u`)

For `b = 1, …, B`:
- Draw a **cluster-level Rademacher weight** `w_g ∈ {−1, +1}` for
  each cluster.
- `y* _g = X_g β̂ + w_g · û_g` — preserves within-cluster
  correlation.
- Refit OLS on `(X, y*)` → `β̂* ᵇ`, `t*ᵇ`.

Percentile / t-percentile CI from the bootstrap distribution.

## Files

- `python/wild_cluster_bootstrap.py` — from-scratch wild-cluster
  bootstrap + CR1 sandwich + naive OLS SEs for comparison. Demo
  (G=12 clusters of 40, β=0.5, heterogeneous cluster shocks): OLS
  SE=0.033 (wrong), CR1 SE=0.035, wild-cluster 95% CI [0.461, 0.589]
  (implied SE 0.033) — CI centred at β̂=0.527 correctly covers truth
  and reflects the clustered dependence.
- `r/wild_cluster_bootstrap.R` — `fwildclusterboot::boottest`,
  `clubSandwich`, `lmtest/sandwich`, `multiwayvcov::cluster.boot`
  (R); `wildboottest` (Python).

## When to use

- **Cluster-heteroskedastic residuals with small G** — DiD with few
  states, education / classroom studies, industry-year cells.
- **Unbalanced clusters** — CR1 SEs get worse; the bootstrap holds.
- **Hypothesis tests with `H_0: β = β_0`** — cluster-restricted
  ("wild bootstrap-c") is size-correct even for very small `G`.

## When NOT to use

- **G large (≥ 40) and balanced** — CR1 sandwich is fine; save
  compute.
- **Non-linear models** — bootstrap can be adapted, but requires
  care with score decomposition.
- **Cross-cluster dependence** — the wild bootstrap needs within-
  cluster resampling; use two-way / spatial variants otherwise.

## Assumptions & caveats

- **Correct clustering variable** — must capture all the important
  correlation.
- **Rademacher vs Mammen weights** — Rademacher (±1) is standard
  for hypothesis testing; Webb-6-point can help with G < 12.
- **Wild-cluster-boot restricted (WCR)** vs unrestricted (WCU) —
  WCR under `H_0` is preferred for size accuracy.
- **Just-passing evidence** — when G is very small (< 6), even wild
  bootstrap over-rejects; consider randomisation inference.

## Related in this repo

- `wild-bootstrap`, `block-bootstrap`, `nonparametric-bootstrap` —
  bootstrap family.
- `newey-west-hac`, `sandwich-robust-se` — asymptotic robust SEs.
- `diff-in-diff`, `staggered-did`, `event-study` — canonical settings
  where wild-cluster boot is standard.

## Run

```
python techniques/wild-cluster-bootstrap/python/wild_cluster_bootstrap.py
Rscript techniques/wild-cluster-bootstrap/r/wild_cluster_bootstrap.R
```

**Refs:** Cameron, A.C., Gelbach, J.B. & Miller, D.L. "Bootstrap-based improvements for inference with clustered errors." *REStat*, 90(3): 414-427, 2008; MacKinnon, J.G. & Webb, M.D. "The wild bootstrap for few (treated) clusters." *Econometrics Journal*, 21(2): 114-135, 2018.

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
