# Sure Independence Screening (Reference §32.7)

Fan & Lv (2008). For `p ≫ n` (millions of features), even LASSO is
intractable. **SIS** pre-screens by marginal correlation with `y`:

```
ω_j = |corr(X_j, y)|,   j = 1..p.
```

Keep the top `d_n` features (typically `n / log n` or `n − 1`). Under
mild conditions, the true active set is retained with probability → 1
(the **sure-screening** property). Iterative SIS (ISIS) alternates
screening with regularised regression on residuals.

## When to use

- **Ultra-high dimensional regression** (`p = 10⁴-10⁸`) — genomic,
  imaging, financial.
- **Pre-processing** step before LASSO / SCAD / knockoffs.
- **Compute-constrained** environments.

## When NOT to use

- **Signals with zero marginal correlation** — SIS misses features
  that only matter jointly; use ISIS or higher-order screening.
- **Correlated predictors** — marginal corr may miss the true signal;
  ISIS + conditional screening (Barut 2016).

## Files

- `python/sure_independence_screening.py` — marginal-correlation
  screening + downstream LASSO comparison. Demo `n=200, p=5000`, 5
  signals: **SIS retains all 5** in top `d_n = 42`; downstream LASSO
  runs 104× faster and cuts false positives from 25 → 6.
- `r/sure_independence_screening.R` — `SIS` (R); `celer` (Python).

## Assumptions & caveats

- **Marginal correlation** — misses interaction-only signals.
- **`d_n` choice** — `⌊n / log n⌋` or `n − 1` (Fan-Lv 2008); larger =
  safer, smaller = more compute savings.
- **ISIS iterations** improve recall in correlated designs.
- **Non-linear extension** — distance-correlation SIS (Li-Zhong-Zhu 2012).

## Related in this repo

- `ridge-lasso-elasticnet`, `adaptive-lasso`, `scad-mcp-penalties`,
  `debiased-lasso`, `model-x-knockoffs`, `stability-selection`,
  `group-lasso` — the high-dim toolbox.
- `partial-least-squares` — supervised dim reduction alternative.

## Run

```
python techniques/sure-independence-screening/python/sure_independence_screening.py
Rscript techniques/sure-independence-screening/r/sure_independence_screening.R
```

**Refs:** Fan, J. & Lv, J. "Sure independence screening for ultrahigh dimensional feature space." *JRSS-B*, 2008; Fan, J., Samworth, R. & Wu, Y. "Ultrahigh dimensional feature selection: beyond the linear model." *JMLR*, 2009.

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
