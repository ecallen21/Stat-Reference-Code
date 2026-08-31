# Debiased LASSO (Reference §32.4)

**Confidence intervals + p-values for LASSO coefficients in `p > n`
regimes.** Zhang & Zhang (2014) and van de Geer et al. (2014).

## Formula

```
β̂_debiased  =  β̂_LASSO  +  (1 / n) · M · Xᵀ (y − X β̂_LASSO)
```

`M ≈ Σ^{-1}` where `Σ = XᵀX / n`. `M` is built via **node-wise LASSO
regressions**: for each column `j`, regress `X_j` on `X_{-j}` with
LASSO to get sparse rows of `M`.

Under mild sparsity + eigenvalue conditions,

```
√n · (β̂_debiased_j − β_true_j)   →   𝒩(0, σ_j²)
```

so a 95 % CI is `β̂_debiased_j ± 1.96 · σ̂_j / √n`.

## When to use

- **High-dim regression** where you need CIs / p-values on individual
  coefficients.
- **Post-LASSO inference** with a formal guarantee.
- **Auditing which selected variables are "really significant"**.

## When NOT to use

- **Very small n / high correlations** — the node-wise LASSOs break
  down; use bootstrap-LASSO or knockoffs.
- **Selection-only tasks** (no CIs needed) — plain LASSO / SCAD is
  cheaper.

## Files

- `python/debiased_lasso.py` — from-scratch coordinate-descent LASSO
  + node-wise LASSO for `M` + Zhang-Zhang correction + Gaussian CI.
  Monte-Carlo over 100 trials (`n = 200, d = 40`, 3 signals):
  **95 % nominal → empirical 90-98 % coverage on signals; 95.2 %
  coverage on 37 zeros; average CI width ≈ 0.15**.
- `r/debiased_lasso.R` — `hdi::lasso.proj`, `hdi::boot.lasso.proj`,
  `selectiveInference` (R); `celer`, `hdlasso` (Python).

## Assumptions & caveats

- **Sparsity `s = o(√n / log p)`** — required by the theory for valid
  CIs; violated in dense regimes.
- **Bounded eigenvalues of `Σ`** — restricted-eigenvalue condition.
- **Tuning `λ` and `λ_M`** — separate CV for the LASSO and the
  node-wise LASSOs; the demo uses fixed values for compactness.
- **Coverage under correlation** — CI widths inflate when features
  correlate strongly (`M` becomes noisy).
- **Alternatives** — `sqrt-LASSO`, `desparsified-LASSO`,
  `selectiveInference` (post-selection conditional CIs), `knockoffs`.

## Related in this repo

- `ridge-lasso-elasticnet`, `adaptive-lasso`, `scad-mcp-penalties` —
  the sparse-fitting family.
- `model-x-knockoffs`, `stability-selection` — FDR-controlled
  selection alternatives.
- `sandwich-robust-se` — heteroscedastic-consistent SEs (low-dim
  cousin).
- `bootstrap`, `bca-bootstrap` — resampling-based SEs.

## Run

```
python techniques/debiased-lasso/python/debiased_lasso.py
Rscript techniques/debiased-lasso/r/debiased_lasso.R
```

**Refs:** Zhang, C.-H. & Zhang, S. "Confidence intervals for low-dimensional parameters in high-dimensional linear models." *JRSS-B*, 2014; van de Geer, S. et al. "On asymptotically optimal confidence regions and tests for high-dimensional models." *Annals of Statistics*, 2014.

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
