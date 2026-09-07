# U-Statistics (Reference §46.9)

Hoeffding (1948). A U-statistic is an unbiased estimator of
`θ = E[h(X₁, …, X_m)]` built by averaging a symmetric kernel `h` of
order `m` over all `m`-tuples from the sample:

    U_n = 1 / C(n, m) · Σ_{i₁<…<i_m} h(X_{i₁}, …, X_{i_m})

## Classical examples

| Order | Kernel | Estimand |
|---|---|---|
| 1 | `h(x) = x` | Sample mean |
| 2 | `h(x, y) = (x − y)² / 2` | Sample variance (unbiased) |
| 2 | `h(x, y) = |x − y| / 2` | Half Gini mean difference |
| 2 | `h(x, y) = sgn((X₁ − X₂)(Y₁ − Y₂))` | Kendall's τ |
| 3 | `h(x, y, z) = 1[y between x, z]` | Bergsma-Dassios tests of trend |

## Asymptotics (Hoeffding)

    √n (U_n − θ) →d N(0, m² · σ₁²)
    σ₁² = Var(h_1(X_1)),    h_1(x) = E[h(x, X_2, …, X_m)]

## Files

- `python/u_statistics.py` — U-statistic implementations (mean,
  variance, Gini, exact combinatorial), Hoeffding projection variance
  via Monte Carlo, and a bootstrap variance check. Demo (n=400, N(0,1)):
  U-mean=−0.037; U-var=0.993; Gini mean diff = 1.120 (truth 1.128 =
  2/√π); Hoeffding SE(U_var) ≈ 0.077 vs bootstrap SE 0.072.
- `r/u_statistics.R` — `Ustat`, `ineq`, `Kendall`, `ape::mantel.test`
  (R); from-scratch (Python).

## When to use

- **Constructing unbiased estimators** from a symmetric kernel — the
  standard route to variance / Gini / Kendall / tests of trend.
- **Deriving asymptotic variance** for rank-based tests.
- **Theoretical proofs** — CLT, LIL, and V-statistic bounds hinge
  on the U-statistic representation.

## When NOT to use

- **High-order kernels with large n** — exact enumeration is
  `O(n^m)`; use random subsampling (V-statistic) or projection
  formulas.
- **Non-iid data** — the classical theory needs iid; use
  network-U or blocked-U variants for dependence.

## Assumptions & caveats

- **iid sample** — required for Hoeffding's CLT and variance
  decomposition.
- **Degenerate U-statistics** (`σ₁² = 0`) — the CLT scales as `n`,
  not `√n`; the limit is a weighted χ² not a normal.
- **Bias corrections** — V-statistics (with replacement in the
  kernel) are biased at `O(n⁻¹)`; U removes this.
- **Complexity** — exact m=2 is `O(n²)`; some kernels admit `O(n log n)`
  algorithms (e.g. Gini via sort trick, shown in the demo).

## Related in this repo

- `mann-whitney`, `kendalls-tau`, `spearman-rank-correlation` —
  well-known rank-based U-statistics.
- `nonparametric-bootstrap` — variance / CI for U-statistics.
- `hoeffding-inequality` (see `concentration-inequalities` planning) —
  companion theorem for bounds.

## Run

```
python techniques/u-statistics/python/u_statistics.py
Rscript techniques/u-statistics/r/u_statistics.R
```

**Refs:** Hoeffding, W. "A class of statistics with asymptotically normal distribution." *Ann. Math. Stat.*, 19(3): 293-325, 1948; Serfling, R.J. *Approximation Theorems of Mathematical Statistics*, Wiley, 1980.

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
